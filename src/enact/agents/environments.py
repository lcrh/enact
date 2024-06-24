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

from enact.agents import node_api

import enact

# Data representation
DataT = TypeVar('DataT')
DataTypeT = TypeVar('DataTypeT')


@enact.contexts.register
class Environment(Generic[DataTypeT, DataT], enact.contexts.Context):
  """Provides low level access to an agent environment."""

  @property
  def node_schema(self) -> node_api.NodeSchema:
    """The node schema of the environment."""
    raise NotImplementedError()

  async def post(
    self,
    node_id: node_api.NodeId,
    data: DataT) -> DataT:
    """Execute a post request on the server."""
    raise NotImplementedError()

  async def get(
    self,
    node_id: node_api.NodeId,
    data: DataT) -> DataT:
    """Execute a get request on the server."""
    raise NotImplementedError()

  async def list(self, node_id: node_api.NodeId) -> List[node_api.NodeId]:
    """Execute a list request on the server."""
    raise NotImplementedError()

