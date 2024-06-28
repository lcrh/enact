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
"""An API for interacting with nodes and inspecting node hierarchies.

The goal of this API is to provide an abstraction layer for accessing and
reflecting on dynamic client interfaces.

A node API allows three core operations:
1) get: Get the data of a node without changing world state. Repeated gets
     should sample from the same distribution (or return the same result).
2) post: Post data to a node, potentially changing global state.
3) list: List the elements of a node.

Each of these operations accepts a node ID. Node IDs are path-like strings
which can contain '.' and '/' separators. Examples:

  "foo" - A node with name foo.
  "foo.bar" - The attribute bar of the node foo.
  "foo/bar" - The element bar of the node foo.
  "foo.bar/baz" - The element baz of the attribute bar of the node foo.

Attributes, indicated by '.', are part of the schema of a node, that is,
the set of attributes of a node is fixed. Elements, indicated by '/', on the
other hand are dynamic. Their type is fixed by the schema, but their node IDs
can only be determined via a list call.

Node APIs are built to allow for easy dynamic creation:

Example API definition:
  root = Node()
  foo = root.add_attribute('foo')

  def sync_get_handler(node_id: NodeId, future: Future):
    future.set_result('hello')
  foo.set_get_handler(sync_get_handler, str)

  # Example use:
  future = create_future()
  foo.get(NodeId('foo'), future)
  assert future.result() == 'hello'
"""

import abc
import copy
import dataclasses
import enum
import functools
import re
import textwrap
from typing import (
  Any, Awaitable, Callable, Dict, Generic, Iterable, List, Mapping, Optional,
  Tuple, Type, TypeVar, Union, cast)


import enact


# Data representation
DataT = TypeVar('DataT')


class ComponentType(enum.Enum):
  """The type of component of a node."""
  ROOT = 0
  ATTRIBUTE = 1
  ELEMENT = 2


# Matches a node ID with one more alphanumeric components separated by . or /.
_VALID_REGEX = re.compile(r'([a-zA-Z0-9_\-]+([\./][a-zA-Z0-9_\-]+)*)?')
# Like above, but allows components to be '*'
_VALID_PATTERN_REGEX = re.compile(
  r'((\*|[a-zA-Z0-9_\-])+([\./](\*|[a-zA-Z0-9_\-]+))*)?')
# An individual component, either the first component or one preceded by . or /.
_COMPONENT_REGEX = re.compile(r'(^|[\./])[\*a-zA-Z0-9_\-]+')
_ALNUM_REGEX = re.compile(r'^[a-zA-Z0-9_\-]+$')


class NodeIdTypeError(ValueError):
  """Raised when a Node ID does not have the expected type."""


NodeIdT = TypeVar('NodeIdT', bound='NodeId')
NodeIdLike = Union['NodeId', str]


class Future(Generic[DataT], abc.ABC):
  """A future-style interface."""

  @abc.abstractmethod
  def set_result(self, result: DataT):
    """Set the result of the future."""
    pass

  @abc.abstractmethod
  def set_exception(self, exception: Exception):
    """Set the exception of the future"""
    pass

  @abc.abstractmethod
  async def result(self) -> DataT:
    """Get the result of the future."""

  @abc.abstractmethod
  def done(self) -> bool:
    """Check if the future is done."""



@enact.register
@dataclasses.dataclass(frozen=True)
class NodeId(enact.ImmutableResource):
  """A node ID, which represents a location in a node API schema.

  Examples:
    "" -> The root node
    "foo.bar" -> The attribute bar of the attribute foo of the root node.
    "foo/baz" -> The element baz of the attribute foo of the root node.
  """
  id_str: str

  def cast_to(self, node_id_type: Type[NodeIdT]) -> NodeIdT:
    """Cast a NodeId to a specific subtype."""
    return node_id_type(str(self))

  def try_cast_to(self, node_id_type: Type[NodeIdT]) -> Optional[NodeIdT]:
    """Cast a NodeId to a specific subtype or return None."""
    try:
      return self.cast_to(node_id_type)
    except NodeIdTypeError:
      return None

  @classmethod
  def from_id(cls: Type[NodeIdT], node_id: 'NodeId') -> NodeIdT:
    """Create a new node ID from an existing node ID."""
    # We could just do a constructor, but custom constructors for
    # immutable dataclasses are a bit messy.
    return cls(str(node_id))

  def __post_init__(self):
    if not _VALID_REGEX.fullmatch(self.id_str):
      raise ValueError(f'Invalid node ID "{self}".')
    type_pattern = self.type_pattern()
    if type_pattern:
      self.assert_matches(type_pattern)

  def __repr__(self) -> str:
    """Return a string representation of the node ID."""
    return f'<NodeId {str(self)}>'

  def __str__(self) -> str:
    """Return node ID string."""
    return self.id_str

  def __getitem__(self, i: Any) -> str:
    """Access ID string."""
    return self.id_str[i]

  def __len__(self) -> int:
    """Return length of id string"""
    return len(self.id_str)

  def __eq__(self, other: Any) -> bool:
    if isinstance(other, str):
      return self.id_str == other
    elif isinstance(other, NodeId):
      return self.id_str == other.id_str
    return False

  def __hash__(self) -> int:
    return hash(self.id_str)

  @classmethod
  def type_pattern(cls) -> Optional[str]:
    """Subclasses can add type restrictions."""
    return None

  def rfind(self, *args: Any, **kwargs: Any) -> int:
    """Access to underlying string rfind."""
    return self.id_str.rfind(*args, **kwargs)

  def find(self, *args: Any, **kwargs: Any) -> int:
    """Access to underlying string find."""
    return self.id_str.find(*args, **kwargs)

  def is_attribute(self) -> bool:
    """Return whether the node is an attribute node."""
    collection_delim = self.rfind('/')
    attribute_delim = self.rfind('.')
    return attribute_delim >= collection_delim

  def is_element(self) -> bool:
    """Return whether the node is an element node."""
    collection_delim = self.rfind('/')
    attribute_delim = self.rfind('.')
    return attribute_delim < collection_delim

  @property
  def name(self) -> str:
    """The last component of the node ID."""
    start_index = max(self.rfind('/'), self.rfind('.'))
    return self[start_index + 1:]

  @property
  def parent(self) -> Optional['NodeId']:
    """The parent node ID."""
    start_index = max(
      self.rfind('/'),
      self.rfind('.'))
    if start_index == -1:
      return None
    return NodeId(self[:start_index])

  @classmethod
  def _str_components(cls, value: str) -> Iterable[Tuple[str, ComponentType]]:
    """Iterate through components of the string."""
    for m in _COMPONENT_REGEX.finditer(value):
      component = m.group()
      if component[0]  == '/':
        yield component[1:], ComponentType.ELEMENT
      elif component[0] == '.':
        yield component[1:], ComponentType.ATTRIBUTE
      else:
        yield component, ComponentType.ROOT

  @functools.cached_property
  def components(self) -> List[Tuple[str, ComponentType]]:
    """Iterate through components of the string."""
    return list(self._str_components(str(self)))

  @functools.cached_property
  def parts(self) -> List[str]:
    """Iterate through parts of the string."""
    return list(p for p, _ in self.components)

  def assert_matches(self, pattern: str) -> Tuple[str,...]:
    """Ensure that the ID matches the pattern and return the components."""
    match = self.matches(pattern)
    if match is None:
      raise NodeIdTypeError(
      f'Expected pattern {pattern}, got {self}')
    return match

  def strip_prefix(self, prefix: 'NodeId') -> Optional['NodeId']:
    """Strip a prefix or return None if prefix is not shared."""
    if not self.id_str.startswith(prefix.id_str):
      return None
    return NodeId(self.id_str[len(prefix.id_str) + 1:])

  def matches(self, pattern: str) -> Optional[Tuple[str,...]]:
    """Check if the node ID matches a pattern with '*' placeholders.

    Args:
      pattern: A pattern with '*' placeholders, e.g.,
        "my_collection/*.my_attribute".

    Returns:
      None if not a match, otherwise the string components of the NodeId as a
      tuple.
    """
    if not _VALID_PATTERN_REGEX.fullmatch(pattern):
      raise ValueError(f'Invalid pattern "{pattern}".')
    components = self.components
    pattern_components = list(self._str_components(pattern))
    if len(components) != len(pattern_components):
      return None
    for component, pattern_component in zip(components, pattern_components):
      if pattern_component[0] == '*':
        if pattern_component[1] != component[1]:
          return None
      elif pattern_component != component:
        return None
    return tuple(component[0] for component in components)

  def attribute(self, name: str) -> 'NodeId':
    return NodeId(f'{self}.{name}')

  def element(self, name: str) -> 'NodeId':
    return NodeId(f'{self}/{name}')


class UnsupportedAction(Exception):
  """Raised if an unsupported node action is requested."""

  def __init__(self, action: str, node_id: NodeId):
    super().__init__(f'Unsupported action "{action}" for node "{node_id}".')



class NodeNotFound(KeyError):
  """Raised if the entity is not found."""

  def __init__(self, node_id: NodeId):
    """Create a new NodeNotFound error."""
    super().__init__(f'Could not find node "{node_id}".')


NodeT = TypeVar('NodeT', bound='NodeSchema')


@enact.register
@dataclasses.dataclass
class Signature(enact.Resource):
  """Signature of a node API call."""
  input_type_descriptor: enact.TypeDescriptor
  output_type_descriptor: enact.TypeDescriptor

  @staticmethod
  def from_types(input_type: Type, output_type: Type):
    return Signature(
      input_type_descriptor=enact.from_python_type(input_type),
      output_type_descriptor=enact.from_python_type(output_type))

  def pformat(self, name: str):
    return (f'{name}({self.input_type_descriptor.pformat()}) -> '
            f'{self.output_type_descriptor.pformat()}')


@enact.register
@dataclasses.dataclass
class NodeSchema(Generic[DataT], enact.Resource):
  """A node schema that declares a structured, typed API.

  A schema is a recursive structure that is associated with Optional typing
  information for a get, post and list handler, as well as a set of struct-like
  attributes (which are themselves schemas) and an optional element type, in
  case that the schema describes a collection. Note that a schema can both
  have attributes and describe a collection, e.g.:

    foo -> An attribute foo of the root node.
    /foo -> An element foo of the root node.
    foo.bar -> An attribute bar of the attribute foo of the root node.
    foo/bar -> An element bar of the attribute foo of the root node.

  Attributes of a schema are static can have different types. Elements of a
  schema are dynamic and all have the same type (self.element_type).

  Note that the schema does not provide access to handlers that would actually
  do anything, but simply outlines the type structure of an API. There are two
  subclasses, FutureNode and Node, which add handlers to the core schema
  structure.
  """
  attributes: Mapping[str, 'NodeSchema[DataT]'] = dataclasses.field(
    default_factory=dict)

  post_signature: Optional[Signature] = None

  get_signature: Optional[Signature] = None

  listable: bool = False

  element_type: Optional['NodeSchema[DataT]'] = None

  description: Optional[str] = None

  def pformat(self, name: str='root', indent='  ') -> str:
    """Pretty print this schema."""
    parts = []
    if self.post_signature:
      parts.append(
        textwrap.indent(self.post_signature.pformat('POST'), indent))
    if self.get_signature:
      parts.append(
        textwrap.indent(self.get_signature.pformat('GET'), indent))
    if self.listable:
      textwrap.indent('LIST()', indent)
    for k, v in self.attributes.items():
      parts.append(textwrap.indent(v.pformat(f'.{k}', indent), indent))
    if self.element_type:
      parts.append(
        textwrap.indent(self.element_type.pformat('/*'), indent))
    if not parts:
      return f'{name}'
    return '\n'.join([f'{name}'] + parts)

  def add_attribute(self: NodeT,
                    name: str,
                    node: Optional[NodeT]=None) -> NodeT:
    """Adds an attribute to the node or returns existing."""
    assert _ALNUM_REGEX.match(name), (
      f'Attribute names must be alphanumeric: {name}')
    if name in self.attributes and not node:
      return cast(NodeT, self.attributes[name])
    node = node or self.create()
    self.attributes[name] = node  # type: ignore
    return node

  def make_collection(
      self: NodeT,
      element_type: Optional[NodeT]=None) -> NodeT:
    """Make the node into a collection of the given type."""
    element_type = element_type or self.create()
    self.element_type = element_type
    return self.element_type

  def _next_separator(self, node_id: NodeIdLike, cur: int) -> int:
    """Return the location of the  next separator or end of string."""
    next_element_sep = node_id.find('/', cur)
    next_attribute_sep = node_id.find('.', cur)
    if next_element_sep == -1 and next_attribute_sep == -1:
      end = len(node_id)
    elif next_element_sep == -1:
      end = next_attribute_sep
    elif next_attribute_sep == -1:
      end = next_element_sep
    else:
      end = min(next_element_sep, next_attribute_sep)
    return end

  def _find(self: NodeT,
            node_id: NodeIdLike,
            add: bool,
            cur: int) -> NodeT:
    """Find a child node by node_id or pattern."""
    if cur >= len(node_id):
      return self
    if node_id[cur] == '/':
      next_sep = self._next_separator(node_id, cur + 1)
      if not self.element_type:
        if add:
          self.make_collection()
        else:
          raise ValueError(f'Cannot list node {node_id[:next_sep]}: {node_id}')
      assert self.element_type
      # Skip the element_id
      # pylint: disable=protected-access
      return cast(
        NodeT, self.element_type._find(node_id, add, next_sep))
    else:
      if node_id[cur] == '.':
        cur += 1
      next_sep = self._next_separator(node_id, cur)
      attribute = node_id[cur: next_sep]
      if attribute not in self.attributes:
        if add:
          self.add_attribute(attribute)
        else:
          raise ValueError(f'No such attribute "{attribute}": {node_id}"')
      # pylint: disable=protected-access
      return cast(
        NodeT, self.attributes[attribute]._find(node_id, add, next_sep))

  def find(self: NodeT, node_id: NodeIdLike) -> NodeT:
    """Find a child node by node_id or pattern."""
    return self._find(node_id, False, 0)

  def add(self: NodeT, node_id: NodeIdLike) -> NodeT:
    """Find or add a child node by node_id or pattern."""
    return self._find(node_id, True, 0)

  @classmethod
  def create(cls: Type[NodeT]) -> NodeT:
    """Create a new node instance."""
    return cls()

  @classmethod
  def copy_from(cls: Type[NodeT], value: 'NodeSchema') -> NodeT:
    """Copy common fields from another schema, ignoring handlers."""
    attributes: Dict[str, NodeT] = {}
    for k, v in value.attributes.items():
      attributes[k] = cls.copy_from(cast(NodeT, v))
    element_type: Optional[NodeT] = None
    if value.element_type:
      element_type = cls.copy_from(cast(NodeT, value.element_type))
    return cls(
      description=value.description,
      attributes=attributes,
      element_type=element_type,
      listable=value.listable,
      post_signature=copy.deepcopy(value.post_signature),
      get_signature=copy.deepcopy(value.get_signature))

  def as_future_node(self) -> 'FutureNode[DataT]':
    """Convert to a future node, ignoring handlers."""
    return FutureNode.copy_from(self)  # type: ignore

  def as_async_node(self) -> 'AsyncNode[DataT]':
    """Convert to a future node, ignoring handlers."""
    return AsyncNode.copy_from(self)  # type: ignore

  def as_schema(self) -> 'NodeSchema[DataT]':
    """Convert to a schema, stripping handlers."""
    return NodeSchema.copy_from(self)


def is_subschema(s1: NodeSchema, s2: NodeSchema) -> bool:
  """Returns whether s1 is a subschema of s2.

  A schema s1 is a subschema of s2 if any request supported by s1 is also
  supported by s2.

  Returns:
    True if s1 is a subschema of s2.
  """
  if s1.get_signature and s1.get_signature != s2.get_signature:
    return False
  if s1.post_signature and s1.post_signature != s2.post_signature:
    return False
  if s1.listable and not s2.listable:
    return False
  for k, v in s1.attributes.items():
    if k not in s2.attributes:
      return False
    if not is_subschema(v, s2.attributes[k]):
      return False
  if s1.element_type:
    if not s2.element_type:
      return False
    return is_subschema(s1.element_type, s2.element_type)
  return True


FutureListHandler = Callable[['NodeId', Future[List[NodeId]]], None]
FutureHandler = Callable[['NodeId', Future, Any], None]

ListHandler = Callable[['NodeId'], Awaitable[List[NodeId]]]
Handler = Callable[['NodeId', Any], Awaitable[Any]]


@enact.register
@dataclasses.dataclass
class FutureNode(NodeSchema[DataT]):
  """A node that extends a schema with future-based handlers."""
  attributes: Dict[str, 'FutureNode[DataT]'] = dataclasses.field(
    default_factory=dict)

  post_handler: Optional[FutureHandler] = None

  get_handler: Optional[FutureHandler] = None

  list_handler: Optional[FutureListHandler] = None

  def set_list_handler(
      self,
      handler: FutureListHandler):
    """Set the handler for the list call."""
    assert not self.list_handler
    assert self.element_type, ('Not a collection. Call make_collection.')
    self.list_handler = handler
    self.listable = True

  def set_post_handler(
      self,
      handler: FutureHandler,
      signature: Signature):
    """Set a sync handler for the post call."""
    assert not self.post_handler
    self.post_handler = handler
    self.post_signature = signature

  def set_get_handler(
      self,
      handler: FutureHandler,
      signature: Signature):
    """Set a sync handler for the get call."""
    assert not self.get_handler
    self.get_handler = handler
    self.get_signature = signature

  def get(self, node_id: NodeId, future: Future, arg: DataT):
    """Get results from a node."""
    node = self.find(node_id)
    if not node.get_handler:
      raise UnsupportedAction('get', node_id)
    handler = node.get_handler
    if not node.get_signature:
      raise ValueError(f'No signature for {node_id}')
    handler(node_id, future, arg)

  def post(self, node_id: NodeId, future: Future, arg: DataT):
    """Post data to a node."""
    node = self.find(node_id)
    if not node.post_handler:
      raise UnsupportedAction('post', node_id)
    handler = node.post_handler
    if not node.post_signature:
      raise ValueError(f'No signature for {node_id}')
    handler(node_id, future, arg)

  def list(self, node_id: NodeId, future: Future):
    """List node elements."""
    node = self.find(node_id)
    if not node.list_handler:
      raise UnsupportedAction('list', node_id)
    handler = node.list_handler
    handler(node_id, future)


@enact.register
@dataclasses.dataclass
class AsyncNode(NodeSchema[DataT]):
  """A schema extended with async handlers."""
  attributes: Dict[str, 'AsyncNode[DataT]'] = dataclasses.field(
    default_factory=dict)

  post_handler: Optional[Handler] = None

  get_handler: Optional[Handler] = None

  list_handler: Optional[ListHandler] = None

  def set_list_handler(
      self,
      handler: ListHandler):
    """Set the handler for the list call."""
    assert not self.list_handler
    assert self.element_type, ('Not a collection. Call make_collection.')
    self.list_handler = handler
    self.listable = True

  def set_post_handler(
      self,
      handler: Handler,
      signature: Signature):
    """Set a sync handler for the post call."""
    assert not self.post_handler
    self.post_handler = handler
    self.post_signature = signature

  def set_get_handler(
      self,
      handler: Handler,
      signature: Signature):
    """Set a sync handler for the get call."""
    assert not self.get_handler
    self.get_handler = handler
    self.get_signature = signature

  async def get(self, node_id: NodeId, arg: DataT) -> DataT:
    """Get results from a node."""
    node = self.find(node_id)
    if not node.get_handler:
      raise UnsupportedAction('get', node_id)
    handler = node.get_handler
    if not node.get_signature:
      raise ValueError(f'No signature for {node_id}')
    return await handler(node_id, arg)

  async def post(self, node_id: NodeId, arg: DataT) -> DataT:
    """Post data to a node."""
    node = self.find(node_id)
    if not node.post_handler:
      raise UnsupportedAction('post', node_id)
    handler = node.post_handler
    if not node.post_signature:
      raise ValueError(f'No signature for {node_id}')
    return await handler(node_id, arg)

  async def list(self, node_id: NodeId):
    """List node elements."""
    node = self.find(node_id)
    if not node.list_handler:
      raise UnsupportedAction('list', node_id)
    handler = node.list_handler
    return await handler(node_id)
