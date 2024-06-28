"""Module to run a coroutine with control inversion."""

import abc
import contextlib
import dataclasses
from typing import (
  Any, Awaitable, Callable, Coroutine, Dict, Generic, Optional, Tuple, TypeVar,
  Union)

from enact.coroutines import threaded_async


ReturnT = TypeVar('ReturnT')


@dataclasses.dataclass(frozen=True)
class ExecutionRequest(Generic[ReturnT]):
  """A request issued via the client.

  Attributes:
    fun: The function to call.
    args: The arguments to the request.
    kwargs: The keyword arguments to the request.
  """
  fun: Callable[..., ReturnT]
  args: Tuple
  kw_args: Dict[str, Any]


@dataclasses.dataclass(frozen=True)
class _RequestWithFuture(Generic[ReturnT]):
  """A request with a future that needs to be filled in."""
  request: ExecutionRequest[ReturnT]
  future: threaded_async.Future[ReturnT]


class Client:
  """Base class for a control inversion client."""

  def __init__(self, server: 'Server'):
    """Initialize the controller."""
    self._server = server

  async def execute(
      self, fun: Callable[..., Union[ReturnT, Coroutine[Any, Any, ReturnT]]],
      *args, **kwargs) -> ReturnT:
    """Execute a function or coroutine function args using the server."""
    # pylint: disable=protected-access
    request = ExecutionRequest(fun, args, kwargs)
    request_with_future = _RequestWithFuture(
      request, self._server.create_future())
    await self._server._request_queue.put(request_with_future)
    return await request_with_future.future.result()


ServerT = TypeVar('ServerT', bound='Server')


class Server:
  """Performs control inversion between a main thread and background coroutine.

  The control inverter is designed to be run from a main thread within a
  game-style "update" function. A typical use would look as follows:

  ```
  def update(self):
    # Process some requests from the running coroutine.
    while self.should_keep_processing():
      self.server.process()
    # Exit so as not to block main thread and come back later.
  ```
  """

  def __init__(self, queue_size=10):
    """Create a new control inverter.

    Args:
      queue_size: The maximum number of requests to queue up from the
        coroutine on the background thread.
      timeout: The maximum number of seconds to wait for the next request to
        arrive from the coroutine.
    """
    self._runner = threaded_async.AsyncRunner()
    self._queue_size = queue_size
    self._request_queue: threaded_async.Queue[_RequestWithFuture] = (
      threaded_async.Queue(
        maxsize=queue_size, runner=self._runner))
    self._exit_stack = contextlib.ExitStack()

  def set_event_loop_timeout(self, timeout: Optional[float]):
    """Set default timeout for waiting on tasks on the event loop.

    Args:
      timeout: The timeout in seconds to wait for the event loop to finish
        processing background tasks. If None, will block indefinitely.
    """
    self._runner.call_in_loop_timeout = timeout

  def set_event_loop_exit_timeout(self, timeout: Optional[float]):
    """Set timeout while waiting for the event loop to shut down.

    Args:
      timeout: The timeout in seconds to wait for the event loop to shut down.
        If None will wait indefinitely.
    """
    self._runner.call_in_loop_timeout = timeout

  @abc.abstractmethod
  def _handle_request(self,
                     request: ExecutionRequest[ReturnT],
                     future: threaded_async.Future[ReturnT]):
    """Request handler that needs to be implemented by subclasses.

    Args:
      request: The request to handle.
      future: A future object where the result should be stored.
    """
    raise NotImplementedError()

  def create_future(self) -> threaded_async.Future:
    """Create a new future on the runner."""
    return threaded_async.Future(runner=self._runner)

  def create_background_task(self, awaitable: Awaitable[ReturnT]) -> (
      threaded_async.BackgroundTask[ReturnT]):
    """Create a task from an awaitable."""
    task = self._runner.create_task(awaitable)
    # Wait for the event loop to be caught up, to ensure that any requests
    # issued by the task have been pushed onto the queue.
    self.wait_on_eventloop()
    return task

  def wait_on_eventloop(self):
    """Wait for the next eventloop iteration."""
    self._runner.wait_for_next_eventloop_iteration()

  def create_client(self) -> Client:
    """Create a client connected to this server."""
    return Client(self)

  def process(self, timeout: Optional[float] = None) -> bool:
    """Wait for the next request and call the handler.

    Note that this will block the main thread in cases where the event loop
    is busy. The upside of this is that it will make sure the event-loop is
    caught up with the current event-loop iteration.

    Args:
      timeout: An optional timeout in seconds to wait for the next request.

    Returns:
      Whether there was a request to process.

    Raises:
      TimeoutError: If the next request could not be finished before the
        timeout was complete.
    """
    if self._request_queue.empty():  # Blocks if the event loop is busy.
      return False
    try:
      request_with_future = self._request_queue.get_wait(timeout=timeout)
    except TimeoutError as e:
      raise TimeoutError(
        'Request queue timed out despite queue being non-empty.') from e
    self._handle_request(
      request_with_future.request,
      request_with_future.future)
    return True

  def __enter__(self: ServerT) -> ServerT:
    """Start the server."""
    self._exit_stack.enter_context(self._runner)
    # Force queue initialization at startup.
    self._request_queue.empty()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Exit the context."""
    self._exit_stack.close()
