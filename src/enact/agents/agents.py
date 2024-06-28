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
"""Top level definition of a software agent."""

import dataclasses
import pprint
from typing import List, TypeVar

import enact
from enact.agents import environments
from enact.agents import node_api


DataT = TypeVar('DataT')

@enact.register
@dataclasses.dataclass
class Agent(enact.Resource):
  """An agent, defined as a node API that executes against an environment.

  Attributes:
    environment_schema: The node schema for the environment the agent operates
      in. The environment is accessed by the agent_node handlers. These handlers
      expect that they are executed in the context of an appropriate environment
      client, that is, that the environment client is accessible via
      environments.Environment.current().
      The environment should expose passive observations as get handlers and
      actions or 'active observations' that may modify the environment as post
      handlers.
    agent_node: The node that represents the agent's functionality. This may
      be a single root_node with handlers or it may be a more complex hierarchy
      of functions that the agent can execute. Note that the agent's get
      handlers should not access the environment's post handlers, as a get
      handler guarantees that the environment is not changed whereas a post
      handler does not.
  """
  name: str
  environment_schema: node_api.NodeSchema[DataT]
  agent_handlers: node_api.AsyncNode[DataT]

  def _check_environment(self, environment: environments.Environment[DataT]):
    """Check that the environment is compatible with the agent."""
    if not node_api.is_subschema(
        self.environment_schema, environment.node_schema):
      raise ValueError(
        f'Environment schema does not match agent schema: '
        f'\n{pprint.pformat(environment.node_schema)}'
        f'\nvs.'
        f'\n{pprint.pformat(self.environment_schema)}')

  async def post(
      self,
      environment: environments.Environment[DataT],
      node_id: node_api.NodeIdLike,
      data: DataT) -> DataT:
    """Issue a post request to the agent against the specified environment."""
    self._check_environment(environment)
    with environment:
      return await self.agent_handlers.post(node_api.NodeId(node_id), data)

  async def get(
      self,
      environment: environments.Environment[DataT],
      node_id: node_api.NodeIdLike,
      data: DataT) -> DataT:
    """Issue a get request to the agent against the specified environment."""
    self._check_environment(environment)
    with environment:
      return await self.agent_handlers.get(node_api.NodeId(node_id), data)

  async def list(
      self,
      environment: environments.Environment[DataT],
      node_id: node_api.NodeIdLike) -> List[node_api.NodeId]:
    """Issue a list request to the agent against the specified environment."""
    self._check_environment(environment)
    with environment:
      return await self.agent_handlers.list(node_api.NodeId(node_id))
