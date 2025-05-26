# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for evaluating programs proposed by the Sampler."""
import ast
import time
from collections.abc import Sequence
import copy
from typing import Any
import signal

from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_globals, safe_builtins, guarded_unpack_sequence, full_write_guard
from RestrictedPython.transformer import ALLOWED_FUNC_NAMES

from funsearch.implementation import code_manipulation
from funsearch.implementation import programs_database


class _FunctionLineVisitor(ast.NodeVisitor):
  """Visitor that finds the last line number of a function with a given name."""

  def __init__(self, target_function_name: str) -> None:
    self._target_function_name: str = target_function_name
    self._function_end_line: int | None = None

  def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
    """Collects the end line number of the target function."""
    if node.name == self._target_function_name:
      self._function_end_line = node.end_lineno
    self.generic_visit(node)

  @property
  def function_end_line(self) -> int:
    """Line number of the final line of function `target_function_name`."""
    assert self._function_end_line is not None  # Check internal correctness.
    return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
  """Extracts the body of the generated function, trimming anything after it."""
  if not generated_code:
    return ''
  code = f'def fake_function_header():\n{generated_code}'
  tree = None
  # We keep trying and deleting code from the end until the parser succeeds.
  while tree is None:
    try:
      tree = ast.parse(code)
    except SyntaxError as e:
      code = '\n'.join(code.splitlines()[:e.lineno - 1])
  if not code:
    # Nothing could be saved from `generated_code`
    return ''

  visitor = _FunctionLineVisitor('fake_function_header')
  visitor.visit(tree)
  body_lines = code.splitlines()[1:visitor.function_end_line]
 
  return '\n'.join(body_lines) + '\n\n'


def _sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
  """Returns the compiled generated function and the full runnable program."""
  #body = _trim_function_body(generated_code)
  body = generated_code
  if version_generated is not None:
    body = code_manipulation.rename_function_calls(
        body,
        f'{function_to_evolve}_v{version_generated}',
        function_to_evolve)
  program = copy.deepcopy(template)
  evolved_function = program.get_function(function_to_evolve)
  evolved_function.body = body
  return evolved_function, str(program)


class Sandbox:
  """Sandbox for executing generated code."""

  def run(
      self,
      program: str,
      function_to_run: str,
      test_input: str,
      timeout_seconds: int,
  ) -> tuple[Any, bool]:
    """Returns `function_to_run(test_input)` and whether execution succeeded."""
    def timeout_handler(signum, frame):
      pass
      #raise TimeoutError("Execution timed out")
    
    try:
      # Set up timeout
      signal.signal(signal.SIGALRM, timeout_handler)
      signal.alarm(timeout_seconds)
      
      # Compile with RestrictedPython
      byte_code = compile_restricted(program, filename="<sandbox>", mode="exec")
      if byte_code is None:
        return None, False
      
      # Create safe execution environment
      safe_builtins_dict = safe_builtins.copy()
      safe_builtins_dict['__import__'] = __import__
      
      # Create dummy funsearch module with decorators
      class FunSearchModule:
        @staticmethod
        def run(func):
          """Dummy decorator for @funsearch.run"""
          return func
        
        @staticmethod
        def evolve(func):
          """Dummy decorator for @funsearch.evolve"""
          return func
      
      restricted_globals = {
        '__builtins__': safe_builtins_dict,
        '__name__': '__main__',
        'funsearch': FunSearchModule(),
        'time': time,
        '_getiter_': iter,
        '_getattr_': getattr,
        '_iter_unpack_sequence_': iter,
        '_getitem_': lambda obj, key: obj[key],
        '_unpack_sequence_': guarded_unpack_sequence,
        '_write_': full_write_guard,
        'range': range,
        'len': len,
        'max': max,
        'min': min,
        'sum': sum,
        'abs': abs,
        'int': int,
        'float': float,
        'str': str,
        'list': list,
        'dict': dict,
        'set': set,
        'tuple': tuple,
      }
      
      local_namespace = {}
      
      # Execute the program
      exec(byte_code, restricted_globals)

      # Get and run the target function
      if function_to_run not in restricted_globals:
        return None, False
      print(f"about to evaluate the following program\n{program}")
      func = restricted_globals[function_to_run]
      result = func(test_input)
      
      return result, True
      
    except Exception as e:
      print(e)
      return None, False
    finally:
      signal.alarm(0)  # Cancel the alarm


def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
  """Returns whether the generated function is calling an earlier version."""
  for name in code_manipulation.get_functions_called(program):
    # In `program` passed into this function the most recently generated
    # function has already been renamed to `function_to_evolve` (wihout the
    # suffix). Therefore any function call starting with `function_to_evolve_v`
    # is a call to an ancestor function.
    if name.startswith(f'{function_to_evolve}_v'):
      return True
  return False


class Evaluator:
  """Class that analyses functions generated by LLMs."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      template: code_manipulation.Program,
      function_to_evolve: str,
      function_to_run: str,
      inputs: Sequence[Any],
      timeout_seconds: int = 30,
  ):
    self._database = database
    self._template = template
    self._function_to_evolve = function_to_evolve
    self._function_to_run = function_to_run
    self._inputs = inputs
    self._timeout_seconds = timeout_seconds
    self._sandbox = Sandbox()

  def analyse(
      self,
      sample: str,
      island_id: int | None,
      version_generated: int | None,
  ) -> None:
    """Compiles the sample into a program and executes it on test inputs."""
    new_function, program = _sample_to_program(
        sample, version_generated, self._template, self._function_to_evolve)

    scores_per_test = {}
    for current_input in self._inputs:
      test_output, runs_ok = self._sandbox.run(
          program, self._function_to_run, current_input, self._timeout_seconds)
      if (runs_ok and not _calls_ancestor(program, self._function_to_evolve)
          and test_output is not None):
        if not isinstance(test_output, (int, float)):
          raise ValueError('@function.run did not return an int/float score.')
        scores_per_test[current_input] = test_output
    if scores_per_test:
      self._database.register_program(new_function, island_id, scores_per_test)
