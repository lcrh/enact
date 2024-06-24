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
from typing import Generic, List, Optional, TypeVar

import enact

from enact.agents import environments
from enact.agents import node_api


T = TypeVar('T')

# Data handled by this agent.
DataT = TypeVar('DataT')


@enact.register
@dataclasses.dataclass
class Agent(Generic[DataT], enact.Resource):
  """An agent, defined as a node API that executes against an environment.

  Attributes:
    environment_schema: The node schema for the environment the agent operates
      in. The environment is accessed by the agent_node handlers. These handlers
      expect that they are executed in the context of the environment client,
      that is, that the environment client is accessible via
      node_client.Client.current().
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
  environment_schema: Optional[
    enact.Ref[node_api.NodeSchema[DataT]]]
  agent_node: enact.Ref[node_api.AsyncNode[DataT]]

  async def __call__(
      self,
      environment: environments.Environment,
      *args: DataT, **kwargs: DataT) -> DataT:
    """Run the agent. This requires a post handler to be set at agent root."""
    return await self.post('', environment, *args, **kwargs)

  async def post(
      self,
      node_id_like: node_api.NodeIdLike,
      environment: environments.Environment,
      *args: DataT,
      **kwargs: DataT) -> DataT:
    """Run an agent's post handler at a given ID."""
    if not isinstance(node_id_like, node_api.NodeId):
      node_id_like = node_api.NodeId(node_id_like)
    with environment:
      return await self.agent_node().post(node_id_like, *args, **kwargs)

  async def get(
      self,
      node_id_like: node_api.NodeIdLike,
      environment: environments.Environment,
      *args: DataT,
      **kwargs: DataT) -> DataT:
    """Run an agent's get handler at a given ID."""
    if not isinstance(node_id_like, node_api.NodeId):
      node_id_like = node_api.NodeId(node_id_like)
    with environment:
      return await self.agent_node().get(node_id_like, *args, **kwargs)

  async def list(
      self,
      node_id_like: node_api.NodeIdLike,
      environment: environments.Environment) -> List[node_api.NodeId]:
    """Run an agent's list handler at a given ID."""
    if not isinstance(node_id_like, node_api.NodeId):
      node_id_like = node_api.NodeId(node_id_like)
    with environment:
      return await self.agent_node().list(node_id_like)
