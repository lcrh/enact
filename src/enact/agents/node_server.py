# Copyright 2024 Agentic.AI Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A node server that offers a AsyncNode access to a FutureNode."""

from typing import Generic, List, TypeVar

from enact.agents import node_api
from enact.coroutines import control_inversion
from enact.coroutines.threaded_async import Future


DataT = TypeVar('DataT')


class Server(Generic[DataT], control_inversion.Server):
  """A node server that translates between an AsyncNode and a FutureNode."""

  def __init__(self, future_node: node_api.FutureNode[DataT]):
    """Create a new node server for the given schema."""
    super().__init__()
    self._future_node = future_node
    self._async_node = future_node.as_async_node()
    self._setup_handlers(self._async_node)
    self._client = self.create_client()

  @property
  def async_node(self) -> node_api.AsyncNode[DataT]:
    """Return the async node for this server."""
    return self._async_node

  def _setup_handlers(self, async_node: node_api.AsyncNode[DataT]):
    """Set all handlers indicated by the schema."""
    if async_node.get_signature is not None:
      async_node.get_handler = self._async_get_handler
    if async_node.post_signature is not None:
      async_node.post_handler = self._async_post_handler
    if async_node.listable:
      async_node.list_handler = self._async_list_handler
    for child in async_node.attributes.values():
      self._setup_handlers(child)
    if async_node.element_type is not None:
      self._setup_handlers(async_node.element_type)

  def _handle_request(
      self,
      request: control_inversion.ExecutionRequest,
      future: Future):
    """Handle a control inversion request."""
    if request.fun == Server._async_get_handler:
      node_id, data = request.args
      assert isinstance(node_id, node_api.NodeId)
      self._future_node.get(node_id, future, data)
    elif request.fun == Server._async_post_handler:
      node_id, data = request.args
      assert isinstance(node_id, node_api.NodeId)
      self._future_node.post(node_id, future, data)
    elif request.fun == Server._async_list_handler:
      node_id, = request.args
      assert isinstance(node_id, node_api.NodeId)
      self._future_node.list(node_id, future)

  async def _async_get_handler(
    self, node_id: node_api.NodeId, data: DataT) -> DataT:
    """Handle a get request."""
    return await self._client.execute(Server._async_get_handler, node_id, data)

  async def _async_post_handler(
    self, node_id: node_api.NodeId, data: DataT) -> DataT:
    """Handle a post request."""
    return await self._client.execute(Server._async_post_handler, node_id, data)

  async def _async_list_handler(
    self, node_id: node_api.NodeId) -> List[node_api.NodeId]:
    """Handle a get request."""
    return await self._client.execute(Server._async_list_handler, node_id)
