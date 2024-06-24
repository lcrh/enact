#  Copyright 2023 Agentic.AI Corporation.
"""Module for running async event loops in background threads."""

import abc
import asyncio
import contextlib
import ctypes
import dataclasses
import enum
import logging
import threading
import traceback
import warnings
from concurrent import futures

from typing import (
  Any, Awaitable, Callable, Generic, List, Optional, Sequence, Set, Type,
  TypeVar, cast)


# We only need this import for the futures interface. I played around for a
# a while trying to get structural subtyping to work instead so we could avoid
# this dependency, but ultimately gave up. (leo)
from enact.agents import node_api

R = TypeVar('R')


def _force_thread_stop(
    thread: threading.Thread, exception: Type[BaseException]):
  """Raises an exception on another thread."""
  ident = thread.ident
  if ident is None:
    raise ValueError('Thread has not been started.')
  ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
    ctypes.c_ulong(ident),
    ctypes.py_object(exception))
  if ret == 0:
    raise ValueError(f'Thread not found: id={ident}')
  elif ret > 1:
    raise SystemError('Failed to raise exception on thread.')


_UNAWAITED_COROUTINE_REGEX = 'coroutine .* was never awaited'

@contextlib.contextmanager
def _suppress_unawaited_warning():
  warnings.filterwarnings(
    'ignore', category=RuntimeWarning,
    message=_UNAWAITED_COROUTINE_REGEX)
  try:
    yield
  finally:
    warnings.filterwarnings(
      'default', category=RuntimeWarning,
      message=_UNAWAITED_COROUTINE_REGEX)


class EventLoopTimeoutError(TimeoutError):
  """Raised when waiting on the event loop times out."""


class BackgroundTask(Generic[R], abc.ABC):
  """An async task running in a background thread."""

  @abc.abstractmethod
  def wait(self, timeout: Optional[float]=None) -> R:
    """Block and wait for result.

    Args:
      timeout: An optional timeout.

    Returns:
      The task result.

    Raises:
      EventLoopTimeoutError: If the task does not complete within the timeout.
      Exception: If the task raises an exception.
    """

  @abc.abstractmethod
  def done(self) -> bool:
    """Return whether the task is done."""

  @abc.abstractmethod
  def cancel(self):
    """Cancel the background task."""

@dataclasses.dataclass(frozen=True)
class ExceptionDetails:
  exception: Exception
  traceback: str


class ExceptionGroup(Exception):
  """Group of exceptions raised by background task."""

  def __init__(self):
    """Create a new background task error."""
    super().__init__()
    self._exception_details: List[ExceptionDetails] = []

  @property
  def exception_details(self) -> Sequence[ExceptionDetails]:
    """The exception details that have been recorded."""
    return tuple(self._exception_details)

  @property
  def exceptions(self) -> Sequence[Exception]:
    """The exceptions that have been recorded."""
    return tuple(e.exception for e in self.exception_details)

  @property
  def tracebacks(self) -> Sequence[str]:
    """The tracebacks associated with each recorded exception."""
    return tuple(e.traceback for e in self.exception_details)

  def record_exception(self, exception: Exception):
    """Add an exception as it is being raised."""
    self._exception_details.append(ExceptionDetails(
      exception, traceback.format_exc()))

  def __str__(self) -> str:
    """Stringify the combined exception."""
    return '\n'.join(self.tracebacks)


class ShutdownType(enum.IntEnum):
  """Returns how the event loop was shut down."""
  NOT_SHUT_DOWN = 0          # Event loop was not shut down
  STOPPED = 1                # Event loop was stopped normally
  FORCE_STOPPED = 2          # Event loop thread was forced to stop.
  FAILED_FORCE_STOPPED = 3   # Thread may still be running.
  UNKNOWN = 4                # Something else went wrong

  def force_stop_attempted(self) -> bool:
    """Returns True if the event loop was force stopped."""
    return self in (ShutdownType.FORCE_STOPPED,
                    ShutdownType.FAILED_FORCE_STOPPED)


# TODO: Figure out why this behaves differently if inheriting from
# BaseException.
class ForcedThreadShutdown(SystemExit):
  """Raised when the event loop thread is forced to stop."""


class AsyncRunner:
  """Runs an async event loop in a background thread.

  Attributes:
    call_in_loop_timeout: Default timeout for call_in_loop, that is, the maximum
      time the main thread will wait for the event loop when scheduling work in
      the loop. This can be overriden by passing an explicit timeout to the
      call_in_loop method.
    exit_timeout: Timeout when waiting for the event loop to exit. If None
      will block indefinitely.
  """

  def __init__(
      self,
      exit_timeout: Optional[float]=1.0,
      call_in_loop_timeout: Optional[float]=1.0,
      debug: bool=False):
    """Start the event loop in the background.

    Args:
      exit_timeout: Timeout when waiting for the event loop to exit.
      call_in_loop_timeout: Default timeout for call_in_loop. Note that if this
        value is too small it may cause spurious timeouts, e.g., when exiting
        the context stack.
      debug: Whether to run the event loop in debug mode, which will track
        tracebacks for all tasks.
    """
    self.exit_timeout = exit_timeout
    self.call_in_loop_timeout = call_in_loop_timeout
    self._event_loop = asyncio.new_event_loop()
    self._event_loop.set_debug(debug)
    self._thread = threading.Thread(
      target=self._event_loop.run_forever, daemon=True)
    self._done = threading.Event()
    self._tasks: Set[_BackgroundTask] = set()
    self._shutdown_type = ShutdownType.NOT_SHUT_DOWN

  def _register_task(self, task: '_BackgroundTask'):
    """Register a task."""
    self._tasks.add(task)

  def _register_task_done(self, task: '_BackgroundTask'):
    """Register a task."""
    self._tasks.discard(task)

  def __enter__(self) -> 'AsyncRunner':
    """Enter context in which event loop is run."""
    assert not self.is_running
    self._thread.start()
    # Run a dummy task to ensure the event loop is running.
    self.run(asyncio.sleep(0.0), None)
    return self

  def _reraise_pending_bg_tasks(self):
    """Reraises errors from background tasks. """
    combined_exceptions = ExceptionGroup()
    for t in self._tasks:
      try:
        t.reraise()
      except asyncio.CancelledError:
        pass
      except ForcedThreadShutdown:
        pass  # This happens during force shutdown.
      except Exception as e:  # pylint: disable=broad-except
        combined_exceptions.record_exception(e)
    if combined_exceptions.exceptions:
      raise combined_exceptions

  def _finalize_shutdown(self):
    """Attempt to force close the event loop if necessary and clean up."""
    with contextlib.ExitStack() as exit_stack:
      if self._shutdown_type != ShutdownType.STOPPED:
        # Suppress async warnings on force shutdown.
        exit_stack.enter_context(_suppress_unawaited_warning())
      if self._thread.is_alive():
        logging.warning(
          'Event loop is stuck. Force killing thread %s', self._thread)
        # Note, if this is stuck in time.sleep call, then the thread will
        # not raise until execution has resumed.
        self._shutdown_type = ShutdownType.FORCE_STOPPED
        _force_thread_stop(self._thread, ForcedThreadShutdown)
        self._thread.join(timeout=self.exit_timeout)
      if self._thread.is_alive():
        self._shutdown_type = ShutdownType.FAILED_FORCE_STOPPED
        logging.warning('Unable to shut down thread %s', self._thread)
      self._reraise_pending_bg_tasks()
      self._done.set()
      try:
        self._event_loop.close()
      except RuntimeError:
        if not self._shutdown_type.force_stop_attempted():
          raise

  def __exit__(self, exc_type, exc_value, tb):
    """Stops the event loop."""
    # Give the event loop a chance to catch up to finish any pending tasks.
    try:
      try:
        self.wait_for_next_eventloop_iteration()
        for t in self._tasks:
          t.cancel()
        # Cancellation is only guaranteed to take effect after the loop runs its
        # next iteration.
        self.wait_for_next_eventloop_iteration()
        _ = self._event_loop.call_soon_threadsafe(self._event_loop.stop)
        self._thread.join(timeout=self.exit_timeout)
        self._shutdown_type = ShutdownType.STOPPED
      except EventLoopTimeoutError:
        pass
      finally:
        self._finalize_shutdown()
    finally:
      if self._shutdown_type == ShutdownType.NOT_SHUT_DOWN:
        self._shutdown_type = ShutdownType.UNKNOWN

  def _assert_running(self):
    """Assert that the event loop is running."""
    assert self.is_running, 'Event loop is not running.'

  @property
  def shutdown_type(self) -> ShutdownType:
    """Provides information as to whether there was an orderly shutdown."""
    return self._shutdown_type

  @property
  def in_runner_loop(self) -> bool:
    """Returns True if we are inside the async runner event loop."""
    try:
      return asyncio.get_running_loop() == self._event_loop
    except RuntimeError:
      return False

  def call_in_loop(
      self,
      fun: Callable[[], R],
      timeout: Optional[float]=None) -> R:
    """Runs callable in loop and returns result.

    Args:
      fun: The callable to run. To run a function with arguments use
        functools.partial or a lambda expression to bind the argument.
      timeout: An optional timeout in seconds. If not set will use
        the call_in_loop_timeout set in the constructor.
    Returns:
      The function return value.
    Raises:
      EventLoopTimeoutError: If the function does not complete within the
        timeout.
    """
    self._assert_running()
    if self.in_runner_loop:
      return fun()
    tracker: _ExecutionTracker[R] = _ExecutionTracker()
    def _wrapper():
      with tracker.track():
        tracker.set_result(fun())
    _ = self._event_loop.call_soon_threadsafe(_wrapper)
    if timeout is None:
      timeout = self.call_in_loop_timeout
    return tracker.wait(timeout=timeout)

  def create_task(self, awaitable: Awaitable[R]) -> BackgroundTask[R]:
    """Create a background task and return it."""
    self._assert_running()
    task = _BackgroundTask.create(awaitable, self)
    return task

  def run(self, awaitable: Awaitable[R],
          timeout: Optional[float]=None) -> R:
    """Run an awaitable in the event loop, block and return the result."""
    self._assert_running()
    return self.create_task(awaitable).wait(timeout=timeout)

  def wait_for_next_eventloop_iteration(self):
    """Blocks until the next event loop iteration."""
    self.run(asyncio.sleep(0.0), timeout=self.call_in_loop_timeout)

  @property
  def is_running(self) -> bool:
    """Returns True if the loop is alive and not being shut down."""
    return self._thread.is_alive() and not self._done.is_set()


class Event:
  """An asyncio event that can be set from an outside thread."""

  def __init__(self, runner: 'AsyncRunner'):
    """Create an Event that may be used for signaling with runner tasks."""
    super().__init__()
    self._runner = runner
    self._event = runner.call_in_loop(asyncio.Event)

  def set(self):
    """Set the event."""
    self._runner.call_in_loop(self._event.set)

  def is_set(self) -> bool:
    """Returns whether the event is set."""
    return self._runner.call_in_loop(self._event.is_set)

  def wait(self, timeout: Optional[float]=None):
    """Wait for the event from outside the loop."""
    self._runner.run(self._event.wait(), timeout=timeout)

  def as_async_event(self) -> asyncio.Event:
    """Return the asyncio event."""
    return self._event


ValueT = TypeVar('ValueT')


class Future(node_api.Future[ValueT]):
  """A future that can be set / awaited inside or outside the loop."""

  def __init__(self, runner: 'AsyncRunner'):
    """Create a new future."""
    self._concurrent_future: futures.Future = futures.Future()
    self._runner = runner
    self._done_event = Event(runner)

  def set_result(self, result: ValueT):
    """Set the result of the future.

    This call will error if a result or exception has already been set.

    Args:
      result: The value that should be retrieved upon calling
        result or result_wait.
    """
    self._concurrent_future.set_result(result)
    self._done_event.set()

  def set_exception(self, exception: Exception):
    """Set the exception of the future.

    This call will error if a result or exception has already been set.

    Args:
      exception: The exception that should be raised upon calling
        result or result_wait.
    """
    self._concurrent_future.set_exception(exception)
    self._done_event.set()

  def done(self) -> bool:
    """Return whether the future is done."""
    return self._concurrent_future.done()

  async def result(self) -> ValueT:
    """Asynchronously await the future."""
    await self._done_event.as_async_event().wait()
    return self._concurrent_future.result()

  def result_wait(self, timeout: Optional[float]=None) -> ValueT:
    """Block thread until the future is complete.

    Args:
      timeout: An optional timeout.

    Raises:
      EventLoopTimeoutError: If the future does not complete within the timeout.
      Exception: If the future raises an exception.
    """
    task = self._runner.create_task(self.result())
    return task.wait(timeout=timeout)



class Queue(Generic[R]):
  """A queue that can coordinate between an AsyncRunner and thread."""

  def __init__(self, runner: 'AsyncRunner', maxsize: int=0):
    """Initialize the queue."""
    self._runner = runner
    self._maxsize = maxsize
    self._async_queue: Optional[asyncio.Queue] = None
    self._lock = threading.Lock()

  def _init_queue(self):
    with self._lock:
      if not self._async_queue:
        self._async_queue = self._runner.call_in_loop(
          lambda: asyncio.Queue(self._maxsize))

  @property
  def _queue(self) -> asyncio.Queue:
    """If not already created, create the queue."""
    if not self._async_queue:
      self._init_queue()
    assert self._async_queue is not None
    return self._async_queue

  def empty(self) -> bool:
    """Return whether the queue is empty."""
    return self._runner.call_in_loop(self._queue.empty)

  def full(self) -> bool:
    """Return whether the queue is full."""
    return self._runner.call_in_loop(self._queue.full)

  def qsize(self) -> int:
    """Return the size of the queue."""
    return self._runner.call_in_loop(self._queue.qsize)

  async def get(self) -> R:
    """Remove and return an item from the queue."""
    return cast(R, await self._queue.get())

  async def put(self, item: R):
    """Put an item on the queue, waiting until room is available"""
    return await self._queue.put(cast(Any, item))

  def get_wait(self, timeout: Optional[float]=None) -> R:
    """Block, remove and return an item from the queue."""
    task = self._runner.create_task(self.get())
    return task.wait(timeout=timeout)

  def put_wait(self, item: R, timeout: Optional[float]=None):
    """Block and return."""
    task = self._runner.create_task(self.put(item))
    return task.wait(timeout=timeout)


class _BackgroundTask(BackgroundTask[R]):
  """Implementation of background tasks."""

  def __init__(self, awaitable: Awaitable[R], runner: 'AsyncRunner'):
    """Internal constructor. Use the "create" method to create a task."""
    self._awaitable = awaitable
    self._runner = runner
    self._async_task: Optional[asyncio.Task] = None
    self._exec_tracker: _ExecutionTracker[R] = _ExecutionTracker()

  @classmethod
  def create(cls,
             awaitable: Awaitable[R],
             runner: 'AsyncRunner') -> BackgroundTask[R]:
    """Create a background task using the runner and return it."""
    bg_task = cls(awaitable, runner)
    bg_task._async_task = runner.call_in_loop(
      # pylint: disable=protected-access
      lambda: asyncio.create_task(bg_task._run()))
    # pylint: disable=protected-access
    runner._register_task(bg_task)
    return bg_task

  async def _run(self):
    """Await the task result."""
    with self._exec_tracker.track():
      self._exec_tracker.set_result(await self._awaitable)

  def reraise(self):
    """Reraise any exceptions that have occurred on the task."""
    self._exec_tracker.reraise()
    # During force shutdown the exception is sometimes not recorded in the
    # tracker, so we also check the task itself.
    if self._async_task and self._async_task:
      try:
        if e := self._async_task.exception():
          raise e
      except asyncio.InvalidStateError:
        # Tasks can enter invalid states during force shutdowns.
        pass

  def wait(self, timeout: Optional[float]=None) -> R:
    """Block and wait for result."""
    assert self._async_task is not None, (
      'Cannot wait for task that has not been started. '
      'Use the "create" method to create a task.')
    if self._runner.in_runner_loop:
      raise RuntimeError('Cannot wait for task in runner loop.')
    if not self._exec_tracker.is_done and not self._runner.is_running:
      raise RuntimeError('Cannot wait for task when runner is not running.')
    try:
      result = self._exec_tracker.wait(timeout=timeout)
    finally:
      if self._exec_tracker.is_done:
        self._runner._register_task_done(self)  # pylint: disable=protected-access
    return result

  def done(self) -> bool:
    """Return whether the task is done."""
    return self._exec_tracker.is_done

  def cancel(self):
    """Cancel the background task."""
    if self._async_task is None:
      raise RuntimeError(
        'Cannot cancel task that has not been started. '
        'Use the "create" method to create a task.')
    self._runner.call_in_loop(self._async_task.cancel)


class _ExecutionTracker(Generic[R]):
  """Context manager for tracking execution of a function.

  TODO: This can probably be rewritten as a Future.

  Used as follows:

  Thread 1:
    with _ExecTracker() as tracker:
      ... # Do something
      tracker.set_result(result)

  Thread 2:
    result = tracker.wait(timeout=1.0)
  """

  def __init__(self):
    """Create a new tracker."""
    self._done = threading.Event()
    self._result: List[R] = []
    self._result_set = False
    self._exception: Optional[BaseException] = None

  def set_result(self, result: R):
    """Set the result."""
    if self._result_set:
      raise RuntimeError('Result already set.')
    self._result_set = True
    self._result.append(result)

  @contextlib.contextmanager
  def track(self):
    """Context manager for tracking executions."""
    try:
      yield self
    except Exception as e:  # pylint: disable=broad-except
      self._exception = e
    except asyncio.CancelledError as e:
      # Note that CancelledError does not inherit from Exception.
      self._exception = e
    finally:
      self._done.set()

  @property
  def is_done(self) -> bool:
    """Returns True if the function has completed."""
    return self._done.is_set()

  def reraise(self):
    """Non-blocking call to reraise any exceptions raised in the context."""
    if self._exception:
      raise self._exception

  def wait(self, timeout: Optional[float]) -> R:
    """Wait for the result."""
    if not self._done.wait(timeout=timeout):
      raise EventLoopTimeoutError(
        'Timed out waiting for execution to complete.')
    if self._exception is not None:
      raise self._exception
    if not self._result_set:
      raise asyncio.CancelledError()
    return self._result[0]
