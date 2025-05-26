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

"""A single-threaded implementation of the FunSearch pipeline."""
import argparse

from collections.abc import Sequence
from typing import Any

from funsearch.implementation import code_manipulation
from funsearch.implementation import config as config_lib
from funsearch.implementation import evaluator
from funsearch.implementation import programs_database
from funsearch.implementation import sampler


def _extract_function_names(specification: str) -> tuple[str, str]:
  """Returns the name of the function to evolve and of the function to run."""
  run_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'run'))
  if len(run_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.run`.')
  evolve_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'evolve'))
  if len(evolve_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.evolve`.')
  return evolve_functions[0], run_functions[0]


def main(specification: str, inputs: Sequence[Any], config: config_lib.Config):
  """Launches a FunSearch experiment."""
  function_to_evolve, function_to_run = _extract_function_names(specification)

  template = code_manipulation.text_to_program(specification)
  database = programs_database.ProgramsDatabase(
      config.programs_database, template, function_to_evolve)

  evaluators = []
  for _ in range(config.num_evaluators):
    evaluators.append(evaluator.Evaluator(
        database,
        template,
        function_to_evolve,
        function_to_run,
        inputs,
    ))
  # We send the initial implementation to be analysed by one of the evaluators.
  initial = template.get_function(function_to_evolve).body
  evaluators[0].analyse(initial, island_id=None, version_generated=None)

  samplers = [sampler.Sampler(database, evaluators, config.samples_per_prompt)
              for _ in range(config.num_samplers)]

  # Run sampling for a configurable number of iterations
  for iteration in range(config.max_iterations):
    print(f"Iteration {iteration + 1}/{config.max_iterations}")
    for s in samplers:
      s.sample()
  
  # Print the best program found
  best_program = database.get_best_program()
  if best_program:
    print("\n" + "="*50)
    print("BEST PROGRAM FOUND:")
    print("="*50)
    print(best_program)
  else:
    print("\nNo program found.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run FunSearch experiment')
  parser.add_argument('--specification', required=True, 
                     help='Path to specification file')
  parser.add_argument('--inputs', required=True,
                     help='Pipe separated list of inputs (e.g., "12,7|15,10")')
  parser.add_argument('--num_samplers', type=int, default=2,
                     help='Number of samplers (default: 2)')
  parser.add_argument('--num_evaluators', type=int, default=4,
                     help='Number of evaluators (default: 4)')
  parser.add_argument('--samples_per_prompt', type=int, default=4,
                     help='Samples per prompt (default: 4)')
  parser.add_argument('--max_iterations', type=int, default=100,
                     help='Maximum number of iterations (default: 100)')
  
  args = parser.parse_args()
  
  # Read specification file
  with open(args.specification, 'r') as f:
    specification = f.read()
  
  inputs = args.inputs.split("|")
  
  # Create config
  config = config_lib.Config(
    num_samplers=args.num_samplers,
    num_evaluators=args.num_evaluators, 
    samples_per_prompt=args.samples_per_prompt,
    max_iterations=args.max_iterations
  )
  
  print(f"Starting FunSearch with:")
  print(f"  Specification: {args.specification}")
  print(f"  Inputs: {inputs}")
  print(f"  Samplers: {args.num_samplers}")
  print(f"  Evaluators: {args.num_evaluators}")
  print(f"  Samples per prompt: {args.samples_per_prompt}")
  print(f"  Max iterations: {args.max_iterations}")
  
  # Run FunSearch
  main(specification, inputs, config)
