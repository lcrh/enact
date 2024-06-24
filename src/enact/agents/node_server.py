# Copyright 2023 Agentic.AI Corporation.
"""Node server with control inversion."""

from typing import Generic, List, TypeVar

from enact.coroutines import control_inversion
from enact.coroutines import threaded_async
from enact.agents import node_api
from enact.agents import environments

ServerT = TypeVar('ServerT', bound='Server')

# Data representation
DataT = TypeVar('DataT')
OutT = TypeVar('OutT')
InT = TypeVar('InT')


class Server(Generic[DataT], control_inversion.Server):
  """A server that drives a SyncNode api from a control inversion client."""

  def __init__(self, node: node_api.FutureNode[DataT]):
    """Initialize the server with a synchronous node API."""
    super().__init__()
    self._node = node

  def _handle_request(
      self,
      request: control_inversion.ExecutionRequest,
      future: threaded_async.Future):
    """Handle a post, list or get request."""
    assert not request.kw_args
    try:
      if request.fun == Environment.get:
        node_id, data = request.args
        assert isinstance(node_id, node_api.NodeId)
        self._node.get(node_id, future, data)
      elif request.fun == Environment.list:
        node_id, = request.args
        assert isinstance(node_id, node_api.NodeId)
        self._node.list(node_id, future)
      elif request.fun == Environment.post:
        node_id, data = request.args
        assert isinstance(node_id, node_api.NodeId)
        self._node.post(node_id, future, data)
    except Exception as e:  # pylint: disable=broad-except
      if not future.done():
        future.set_exception(e)

  def update(self):
    """Handle server events until no new events are available."""
    self._wait_on_eventloop()
    while self.process():
      self._wait_on_eventloop()


class Environment(environments.Environment[DataT]):
  """Agent environment that uses a control inversion server."""

  def __init__(self, server: Server[DataT]):
    """Create a new game client."""
    self._server = server
    self._ci_client = server.create_client()

  @property
  def node_schema(self) -> node_api.NodeSchema:
    """The schema of the client."""
    # pylint: disable=protected-access
    return self._server._node

  async def post(
      self,
      node_id: node_api.NodeId,
      data: DataT) -> DataT:
    """Execute a post request on the server."""
    return await self._ci_client.execute(Environment.post, node_id, data)

  async def get(
      self,
      node_id: node_api.NodeId,
      data: DataT) -> DataT:
    """Execute a get request on the server."""
    return await self._ci_client.execute(Environment.get, node_id, data)

  async def list(self, node_id: node_api.NodeId) -> List[node_api.NodeId]:
    """Execute a list request on the server."""
    return await self._ci_client.execute(Environment.list, node_id)
