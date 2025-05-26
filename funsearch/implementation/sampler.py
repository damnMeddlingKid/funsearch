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

"""Class for sampling new programs."""
import os
from collections.abc import Collection, Sequence


import numpy as np
from google import genai
from google.genai import types

from funsearch.implementation import evaluator
from funsearch.implementation import programs_database


class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int) -> None:
    self._samples_per_prompt = samples_per_prompt
    self._client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    print(f"LLM PROMPT:\n{prompt}\n" + "="*50)
    response = self._client.models.generate_content(
      model='gemini-2.0-flash-001',
      contents=prompt,
      config=types.GenerateContentConfig(
          system_instruction="""
          You are a professional computer scientist skilled in optimizing programs.
          Here is a python program along with a few versions of a function we would like to optimize. 
          Complete the next version of the function with an implementation that is better than the previous attempts.
          
          <IMPORTANT>
          - Return only the completion of the Python code without any markdown formatting, backticks, or code blocks.
          Do not include ```python or ``` in your response. 

          - Only write function you are being asked to complete.
          dont implement other parts of the code, don't implement other versions of the function.
          </IMPORTANT>
          """,
          temperature=0.3,
      ),
    )
    print(f"LLM RESPONSE:\n{response.text}\n" + "="*50)
    return response.text

  def draw_samples(self, prompt: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      samples_per_prompt: int,
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = LLM(samples_per_prompt)

  def sample(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""
    prompt = self._database.get_prompt()
    samples = self._llm.draw_samples(prompt.code)
    # This loop can be executed in parallel on remote evaluator machines.
    for sample in samples:
      chosen_evaluator = np.random.choice(self._evaluators)
      chosen_evaluator.analyse(
          sample, prompt.island_id, prompt.version_generated)
