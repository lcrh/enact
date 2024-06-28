"""Basic task structure."""

import dataclasses
from typing import Any, Iterable, TypeVar
import uuid

from enact.agents import node_api


T = TypeVar('T')


@dataclasses.dataclass(frozen=True)
class Task:
  """Superclass for tasks."""
  task_id: str
  node_id: node_api.NodeId
  future: node_api.Future

  def set_result(self, result: Iterable[node_api.NodeId]):
    self.future.set_result(result)

  def set_exception(self, exception: Exception):
    self.future.set_exception(exception)

  @staticmethod
  def new_id() -> str:
    return uuid.uuid4().hex


@dataclasses.dataclass(frozen=True)
class ListTask(Task):
  """A get task to be executed on the environment."""


@dataclasses.dataclass(frozen=True)
class GetTask(Task):
  """A get task to be executed on the environment."""
  data: Any


@dataclasses.dataclass(frozen=True)
class PostTask(Task):
  """A task to be executed on the environment."""
  data: Any
