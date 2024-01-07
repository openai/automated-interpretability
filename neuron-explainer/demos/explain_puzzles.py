#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


import os

os.environ["OPENAI_API_KEY"] = "put-key-here"

from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.explanations.explainer import TokenActivationPairExplainer
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.puzzles import PUZZLES_BY_NAME


EXPLAINER_MODEL_NAME = "gpt-4"

explainer = TokenActivationPairExplainer(
    model_name=EXPLAINER_MODEL_NAME,
    prompt_format=PromptFormat.HARMONY_V4,
    max_concurrent=1,
)

for puzzle_name, puzzle in PUZZLES_BY_NAME.items():
    print(f"{puzzle_name=}")
    puzzle_answer = puzzle.explanation
    # Generate an explanation for the puzzle.
    explanations = await explainer.generate_explanations(
        all_activation_records=puzzle.activation_records,
        max_activation=calculate_max_activation(puzzle.activation_records),
        num_samples=1,
    )
    assert len(explanations) == 1
    model_generated_explanation = explanations[0]
    print(f"{model_generated_explanation=}")
    print(f"{puzzle_answer=}\n")


