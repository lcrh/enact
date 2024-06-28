# Copyright 2023 Agentic.AI Corporation.
"""A server that can run agents."""

from typing import Dict, Generic, Iterable, List, TypeVar

from enact.coroutines import threaded_async
from enact.agents import agents, environments
from enact.agents import node_api
from enact.agents import tasks
from enact.agents import node_server

# Data representation
DataT = TypeVar('DataT')
AgentServerT = TypeVar('AgentServerT', bound='AgentServer')


class DuplicateTask(ValueError):
  """Raised when two tasks share the same ID."""


class AgentServer(Generic[DataT]):
  """A server that negotiates between an async agent and a sync environment.

  When an agent is deployed on an agent server, it will run on an event loop
  in a background thread. Requests it makes to the environment will be enqueued
  as tasks on the agent server and can be accessed via the `outstanding_tasks`
  property.

  In order for the agent to continue making progress, task futures must be set
  by the owning process and `update` need to be called to allow the event loop
  to proceed.

  Example:
    with AgentServer(env_schema) as server:
      task = server.post(agent, node_id, data)
      while not task.done():
        server.update()
        handle_tasks(server.outstanding_tasks)
  """

  def __init__(self, environment_schema: node_api.NodeSchema[DataT]):
    """Initialize the server with a synchronous node API."""
    self._schema = environment_schema
    self._environment_node = environment_schema.as_future_node()
    self._setup_handlers(self._environment_node)
    self._node_server = node_server.Server(self._environment_node)
    self._outstanding_tasks: Dict[str, tasks.Task] = {}

  @property
  def schema(self) -> node_api.NodeSchema[DataT]:
    """Return the schema."""
    return self._schema

  @property
  def outstanding_tasks(self) -> Iterable[tasks.Task]:
    """Return outstanding tasks."""
    for task in list(self._outstanding_tasks.values()):
      if task.future.done():
        del self._outstanding_tasks[task.task_id]
      else:
        yield task

  def post(self,
           agent: agents.Agent,
           location: node_api.NodeIdLike,
           agent_node: node_api.NodeIdLike,
           data: DataT) -> threaded_async.BackgroundTask[DataT]:
    """Start processing a post request on an agent."""
    environment = environments.Environment(
      self._node_server.async_node, location)
    return self._node_server.create_background_task(
      agent.post(environment, agent_node, data))

  def get(self,
          agent: agents.Agent,
          location: node_api.NodeIdLike,
          agent_node: node_api.NodeIdLike,
          data: DataT) -> threaded_async.BackgroundTask[DataT]:
    """Start processing a get request on an agent."""
    environment = environments.Environment(
      self._node_server.async_node, location)
    return self._node_server.create_background_task(
      agent.get(environment, agent_node, data))

  def list(self,
           agent: agents.Agent,
           location: node_api.NodeIdLike,
           agent_node: node_api.NodeIdLike) -> threaded_async.BackgroundTask[
             List[node_api.NodeId]]:
    """Start processing a list request on an agent."""
    environment = environments.Environment(
      self._node_server.async_node, location)
    return self._node_server.create_background_task(
      agent.list(environment, agent_node))

  def update(self):
    """Handle server events until no new events are available."""
    self._node_server.wait_on_eventloop()
    while self._node_server.process():
      self._node_server.wait_on_eventloop()

  def _enqueue_task(self, task: tasks.Task):
    """Add the task to the queue."""
    if task.task_id in self._outstanding_tasks:
      raise DuplicateTask(f'Task ID {task.task_id} already exists.')
    self._outstanding_tasks[task.task_id] = task

  def _future_get_handler(
      self,
      node_id: node_api.NodeId,
      future: threaded_async.Future,
      data: DataT):
    """Put the get request on the task queue."""
    self._enqueue_task(
      tasks.GetTask(tasks.Task.new_id(), node_id, future, data))

  def _future_post_handler(
      self,
      node_id: node_api.NodeId,
      future: threaded_async.Future,
      data: DataT):
    """Put the post request on the task queue."""
    self._enqueue_task(
      tasks.PostTask(tasks.Task.new_id(), node_id, future, data))

  def _future_list_handler(
            self,
      node_id: node_api.NodeId,
      future: threaded_async.Future):
    """Put the list request on the task queue."""
    self._enqueue_task(
      tasks.ListTask(tasks.Task.new_id(), node_id, future))

  def __enter__(self: AgentServerT) -> 'AgentServerT':
    """Enter the server context."""
    self._node_server.__enter__()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Exit the server context manager."""
    self._node_server.__exit__(exc_type, exc_value, traceback)

  def _setup_handlers(self, future_node: node_api.FutureNode[DataT]):
    """Set all handlers indicated by the schema."""
    if future_node.get_signature is not None:
      future_node.get_handler = self._future_get_handler
    if future_node.post_signature is not None:
      future_node.post_handler = self._future_post_handler
    if future_node.listable:
      future_node.list_handler = self._future_list_handler
    for child in future_node.attributes.values():
      self._setup_handlers(child)
    if future_node.element_type is not None:
      self._setup_handlers(future_node.element_type)
