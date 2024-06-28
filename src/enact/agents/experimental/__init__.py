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
"""Agents/experimental package."""

class EnvServer:
  """Simulated server for registering environments."""

  def __init__(self):
    """Create a new env server."""
    self.envs = {}

  def heartbeat(self, env_id: str):
    """Called by the environment to stay connected."""
    pass

  def connect_environment(self, env):
    """Register"""

