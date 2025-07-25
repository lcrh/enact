# enact - A Framework for Generative Software.

Enact is an in-development python framework for building generative software,
specifically software that integrates with machine learning models or APIs that
generate distributions of outputs.

## Key features

Enact was created to help developers build generative software. Our focus is on
offering core framework functionality required for developing, inspecting and
improving software that samples from generative subsystems such as LLMs and
diffusion models. Another way to understand enact is as a framework for
working with *scaffolding programs*.

To this end, enact boosts your existing python programs by giving you the
ability to
* recursively track generative executions,
* persist execution data and system parameters in versioned storage,
* explore the space of possible system executions via rewind and replay,
* turn your existing code into asynchronous flows that can halt and
  sample input from external systems or users, and
* orchestrate higher-order generative flows in which AI systems write and run
  software.

Enact represents a first-principles approach to supporting development of
generative software close to the level of the programming language. It is
therefore different from and complementary to libraries that focus on unified
integrations (e.g., API wrappers) or on particular algorithm schemas for
orchestrating generative flows (e.g., chain/tree-of-thought based approaches).

See [why-enact](#why-enact) for a more in-depth discussion on why we chose this
approach. For an example of the kind of software that can be written with
enact, take a look at the [MCTS example](examples/generic_mcts.ipynb), which
builds a *generic* implementation of Monte Carlo Tree Search that directly
explores tree of possible executions of a scaffolding program.


## Installation and overview

Enact is available as a [pypi package](https://pypi.org/project/enact/) and can
be installed via:

```bash
pip install enact
```
### Invocations
Enact recursively tracks inputs and outputs of functions registered with the
framework. A record of an execution is called an `Invocation`, and can be
persisted to a store. This is useful for tracking the behavior of generative
software.

```python
import enact
import random

@enact.register
def roll_die(sides: int) -> int:
  """Roll a die."""
  return random.randint(1, sides)

@enact.register
def roll_sum(num_rolls: int) -> int:
  """Roll dice."""
  return sum(roll_die(6) for _ in range(num_rolls))

with enact.InMemoryStore() as store:
  invocation = enact.invoke(roll_sum, (2,))
  print(enact.invocation_summary(invocation))
```
_Output:_
```
->roll_sum(2) = 7
  ->roll_die(6) = 2
  ->roll_die(6) = 5
```

### Rewind/Replay
Invocations can be rewound and replayed:

```python
# Rewind the last roll and replay it.
with store:
  first_roll = invocation.rewind(1)    # Rewind by one roll.
  print('Partial invocation: ')
  print(enact.invocation_summary(first_roll))
  reroll_second = first_roll.replay()  # Replay the second die roll.
  print('\nReplayed invocation: ')
  print(enact.invocation_summary(reroll_second))
```
_Output:_
```python
Partial invocation:
->roll_sum(2) incomplete
  ->roll_die(6) = 2

Replayed invocation:
->roll_sum(2) = 8
  ->roll_die(6) = 2
  ->roll_die(6) = 6
```

### Human-in-the-loop

In enact, humans are treated as just another generative distribution. Their
input can be sampled using `enact.request_input`, which internally raises an
exception. Since the execution itself is persisted in a store, it can be 
replayed at a later time, replacing the exception with the input, thus 
continuing the execution.

```python
@enact.register
def human_rolls_die():
  """Query the user for a die roll."""
  return enact.request_input(requested_type=int, for_value='Please roll a die.')

@enact.register
def roll_dice_user_flow():
  """Request the number of die to roll and optionally sample die rolls."""
  num_rolls = enact.request_input(requested_type=int, for_value='Total number of rolls?')
  return sum(human_rolls_die() for _ in range(num_rolls))

request_responses = {
  'Total number of rolls?': 3,  # Roll 3 dice.
  'Please roll a die.': 6,      # Humans always roll 6.
}

with store:
  invocation_gen = enact.InvocationGenerator.from_callable(
    roll_dice_user_flow)
  # Process all user input requests in order.
  for input_request in invocation_gen:
    invocation_gen.set_input(request_responses[input_request.for_value()])
  print(enact.invocation_summary(invocation_gen.invocation))
```
_Output:_
```
->roll_dice_user_flow() = 18
  ->RequestInput(requested_type=<class 'int'>, context=None)(Total number of rolls?) = 3
  ->human_rolls_die() = 6
    ->RequestInput(requested_type=<class 'int'>, context=None)(Please roll a die.) = 6
  ->human_rolls_die() = 6
    ->RequestInput(requested_type=<class 'int'>, context=None)(Please roll a die.) = 6
  ->human_rolls_die() = 6
    ->RequestInput(requested_type=<class 'int'>, context=None)(Please roll a die.) = 6
```

Note that enact can also be applied to `async` python programs. The advantage of the
enactive-style of asynchronous input sampling is that the partial execution can be
persisted and resumed at a later time or on a different machine.


### Versioning

Enact components are structurally hashed and persisted in a store. This makes it
possible to access ephemeral configurations of software associated only with a
particular run:

```python
import dataclasses

@enact.register
@dataclasses.dataclass
class DiceRoll(enact.Invokable[None, int]):
  num_rolls: int
  sides_per_die: int

  def call(self) -> int:
    return sum(roll_die(self.sides_per_die) for _ in range(self.num_rolls))

with enact.InMemoryStore() as store:
  dice_roll = DiceRoll(2, 6)
  invocation = enact.invoke(DiceRoll(2, 6))
  # modify the diceroll object
  dice_roll.num_rolls = 3
  dice_roll.sides_per_die = 8
  # Fetch the version associated with the first roll:
  print(enact.invocation_summary(invocation))
  print(f'Dice roll object associated with the invocation: '
        f'{invocation.request().invokable()}')
  print(f'Dice roll object after modification: {dice_roll}')
```
_Output:_
```python
->DiceRoll(num_rolls=2, sides_per_die=6)(None) = 7
  -><function roll_die at 0x7fa6e1e42ee0>(6) = 4
  -><function roll_die at 0x7fa6e1e42ee0>(6) = 3
Dice roll object associated with the invocation: DiceRoll(num_rolls=2, sides_per_die=6)
Dice roll object after modification: DiceRoll(num_rolls=3, sides_per_die=8)
```

## Using enact

Full documentation is work in progress.
See [enact concepts](examples/enact_concepts.ipynb) for more information and
take a look at the [examples](examples/).

### Making your types enact compatible

Enact has built-in support for basic python datatypes such as primitives,
lists, sets, dictionaries and tuples.

If you need support for additional types, you can either define them as
`Resource` dataclasses in the framework or define a wrapper.

#### Defining a resource

```python
import dataclasses

@enact.register
@dataclasses.dataclass
class MyResource(enact.Resource):
  x: int
  y: list = dataclasses.field(default_factory=list)

with store:
  # Commit your resource to the store, obtain a reference.
  ref = enact.commit(MyResource(x=1, y=[2, 3]))
  # Check out your reference.
  print(ref.checkout())  # Equivalent to "print(ref())".
```
_Output:_
```python
MyResource(x=1, y=[2, 3])
```

#### Defining a wrapper
For existing types, it can be more convenient to wrap them rather than to
redefine them as enact `Resource` objects.

This can be accomplished by registering a `TypeWrapper` subclass.

```python
class Die:
  """Non-enact python type."""
  def __init__(self, sides):
    self.sides = sides

  @enact.register
  def roll(self):
    return random.randint(1, self.sides)

@enact.register
@dataclasses.dataclass
class DieWrapper(enact.TypeWrapper[Die]):
  """Wrapper for Die."""
  sides: int

  @classmethod
  def wrapped_type(cls):
    return Die

  @classmethod
  def wrap(cls, value: Die):
    """Translate to enact wrapper."""
    return DieWrapper(value.sides)

  def unwrap(self) -> Die:
    """Translate to basic python type."""
    return Die(self.sides)

with store:
  die = Die(sides=6)
  invocation = enact.invoke(die.roll)
  print(enact.invocation_summary(invocation))
```
_Output:_
```
-><__main__.Die object at 0x7f3464398340>.roll() = 4
```

## Why enact - A manifesto

Generative AI models are transforming the software development process.

Traditional software relies primarily on functional buildings blocks in which
inputs and system state directly determine outputs. In contrast, software
that samples one or more generative subsystems associates each input with
a range of possible outputs.

This small change in emphasis - from functions to conditional distributions -
implies a shift across multiple dimensions of the engineering process,
summarized in the table below.

|        |  Generative   |  Traditional |
| :----: | :-----------: | :----------: |
| Building blocks | Conditional distributions | Deterministic functions |
| Engineering | Improve output distributions | Add features, debug errors |
| Subsystems | Unique | Interchangeable |
| Code sharing | Components | Frameworks |
| Execution traces | Training data | Debugging |
| Interactivity | Within system components | At system boundaries |
| Code vs data | Overlapping | Distinct |

### Building generative software means fitting distributions

Generative AI can be used to quickly sketch out impressive end-to-end
experiences. Evolving such prototypes into production-ready systems can be
non-trivial though, as they may show undesirable behaviors a certain percentage
of the time, may fail to respond correctly to unusual user inputs or may simply
fail to meet a softly defined quality target.

Where traditional engineering focuses on implementing features and fixing bugs,
a generative software engineer must tune the distribution represented by the system
as a whole towards some implicitly or explicitly specified target.

In some cases, tuning generative systems can be achieved directly using machine
learning techniques and frameworks. For instance, systems that consist of a thin
software wrapper around a monolithic machine learning model may directly train
the underlying model to shape system behavior.

In other cases, particularly when running complex generative flows that involve
multiple interacting subcomponents that cannot be easily optimized end-to-end,
gradient-free methods are necessary to improve system performance. These range
from using programmers (either human or synthetic) to elaborate the algorithmic
guard-rails of the generative process, running experiments between API identical
subsystems or finding ideal parameters (e.g., prompt templates) for individual
components.

### Fitting generative software distributions involves program search

In generative software, individual components produce distributions that may be
more or less well-suited to the system's overall goal: An LLM that performs well
on tabular data may perform less well when used as a chat agent. One fine-tuned
diffusion model may produce images that are closer to the target style than
another. This is distinct from the the case of traditional software engineering,
where the choice between two correct, API-identical implementations of a module
is of little importance to the behavior of the system as a whole.

Fitting a piece of a generative software involves optimizing the output
distribution towards some quality target. In some cases this can be achieved
using machine learning techniques such as gradient descent, but in cases where
this is not possible, it involves searching for high performing solutions in the
space of software and software configurations.

This search can be manual or automatic, and it can involve changes to parameters
such as LLM prompts or to the scaffolding program itself, e.g., searching over
multiple candidate responses.

The profusion of base models and possible algorithmic elaborations of base
models suggests that gradient-free search methods will take a central place in
generative software development. This could be as simple as running an
experiment in which one component is compared against another or as complex as
searching the space of software, e.g., using LLM-boosted genetic algorithms.

As algorithms in the AlphaZero family have shown us, a particularly fruitful
approach is combining search with machine learning techniques in such a way
that search improves the reasoning power of the core ML model, whereas the ML
model can be improved with the experience gained through search.

By offering in-depth program telemetry and execution search over scaffolding
programs, enact aims to provide the basic infrastructure to enable these kinds
of approaches.

### Generative software is collaborative

There are many generative AI components that are mutually API-compatible (e.g.,
text-to-image generation, LLMs) but that cannot be directly ranked in terms of
quality since they represent a unique take on the problem domain. The
combination of API compatibility and variation lends itself to a more
evolutionary, collaborative style than traditional software, where shared effort
tends to centralize into fewer, lower-level frameworks.

This new collaborative approach is already evidenced by both widespread sharing
of prompt templates and the existence of public databases of fine-tuned models.

Enact aims to simplify collaboration using a content-addressable memory approach
to storage. A program and its executions can be uniquely identified by its hash,
which provides a shared ontology for collaborative program search.

### Generative software reflects on its executions.

Many popular programming languages offer capacity for reflection, wherein the
structure of code is programmatically interpretable by code. Generative software
additionally requires the ability for code to reflect on code execution.

Sampling a generative model is computationally intensive and provides valuable
information that may be used as system feedback or training data. Since
generative software may include complex, recursive algorithms over generative
components, this suggests a need for not simply tracking inputs and outputs
at system boundaries, but at every level of execution.

This is particularly relevant to higher order generative flows, in which the
output of one generative step structures the execution of the next: Such a
system can use call traces as feedback to improve its output, in the same way
that a human may use a debugger to step through an execution to debug an issue.

Enact provides the concept of invocations, and can support higher-order 
invocations: That is, provide in-depth telemetry for software that reflects on
executions of other software.

### Humans are in the loop

Unlike traditional systems, where user interactions happen at system boundaries,
AI and humans are often equal participants in generative computations. An
example is Reinforcement Learning from Human Feedback (RLHF), where humans
provide feedback on AI output, only to be replaced by an AI model that mimics
their feedback behavior during the training process. In other cases, critical
generative flows (e.g., the drafting of legal documents) may require a human
verification step, without which the system provides subpar results.

The choice between sampling human input and sampling a generative model
involves considerations such as cost and quality, and may be subject to change
during the development of a system. For example, an early deployment of a system
may heavily sample outputs or feedback from human participants, until there is
enough data to bootstrap the target distribution using automated generative
components.

Enact's execution search capablities allow humans to alter or inject data
into program executions: A partial execution that failed due to requiring
user attention can be persisted and replayed at a later time.

### Data is code, code is data

Traditional software systems tend to allow a clear distinction between code
(written by developers) and data (generated by the user and the system). In
generative systems this distinction breaks down: Approaches such as
[AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT) or
[Voyager](https://github.com/MineDojo/Voyager) use generative AI to generate
programs (specified in code or plain text), which in turn may be interpreted by
generative AI systems; prompts for chat-based generative AI could equally be
considered code or data.

## Development and Contributing

The framework is open source and Apache licensed. We are actively looking for
contributors that are excited about the vision. If you're interested in the
project, feel free to reach out.

You can download the source code, report issues and create pull requests at
https://github.com/lcrh/enact.
