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
"""Environment interface for agents."""

from typing import Generic, List, TypeVar

import enact
from enact.agents import node_api

# Data representation
DataT = TypeVar('DataT')


@enact.contexts.register
class Environment(Generic[DataT], enact.contexts.Context):
  """Provides low level access to a location in an agent environment."""

  def __init__(
      self,
      handlers: node_api.AsyncNode,
      location: node_api.NodeIdLike=''):
    """Create an environment from a node API."""
    assert isinstance(handlers, node_api.AsyncNode)
    super().__init__()
    self._location = node_api.NodeId(location)
    self._schema = handlers.as_schema()
    self._handlers = handlers

  @property
  def node_schema(self) -> node_api.NodeSchema[DataT]:
    """The node schema of the environment location."""
    return self._schema.find(self._location)

  def _get_absolute_node_id(self, client_node_id: node_api.NodeIdLike):
    """Return the absolute node id relative to location."""
    if not str(client_node_id):
      return self._location
    return node_api.NodeId(f'{self._location}.{client_node_id}')

  async def post(
    self,
    node_id: node_api.NodeIdLike,
    data: DataT) -> DataT:
    """Execute a post request on the environment at the location."""
    return await self._handlers.post(
      self._get_absolute_node_id(node_id), data)

  async def get(
    self,
    node_id: node_api.NodeIdLike,
    data: DataT) -> DataT:
    """Execute a get request on the environment at the location."""
    return await self._handlers.get(
      self._get_absolute_node_id(node_id), data)

  async def list(self, node_id: node_api.NodeIdLike) -> List[node_api.NodeId]:
    """Execute a list request on the environment at the location."""
    return await self._handlers.list(self._get_absolute_node_id(node_id))
