#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


import os

os.environ["OPENAI_API_KEY"] = "put-key-here"

from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.activations.activations import ActivationRecordSliceParams, load_neuron
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.explainer import TokenActivationPairExplainer
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.scoring import simulate_and_score
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator

EXPLAINER_MODEL_NAME = "gpt-4"
SIMULATOR_MODEL_NAME = "text-davinci-003"


# test_response = await client.make_request(prompt="test 123<|endofprompt|>", max_tokens=2)
# print("Response:", test_response["choices"][0]["text"])

# Load a neuron record.
neuron_record = load_neuron(9, 6236)

# Grab the activation records we'll need.
slice_params = ActivationRecordSliceParams(n_examples_per_split=5)
train_activation_records = neuron_record.train_activation_records(
    activation_record_slice_params=slice_params
)
valid_activation_records = neuron_record.valid_activation_records(
    activation_record_slice_params=slice_params
)

# Generate an explanation for the neuron.
explainer = TokenActivationPairExplainer(
    model_name=EXPLAINER_MODEL_NAME,
    prompt_format=PromptFormat.HARMONY_V4,
    max_concurrent=1,
)
explanations = await explainer.generate_explanations(
    all_activation_records=train_activation_records,
    max_activation=calculate_max_activation(train_activation_records),
    num_samples=1,
)
assert len(explanations) == 1
explanation = explanations[0]
print(f"{explanation=}")

# Simulate and score the explanation.
simulator = UncalibratedNeuronSimulator(
    ExplanationNeuronSimulator(
        SIMULATOR_MODEL_NAME,
        explanation,
        max_concurrent=1,
        prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
    )
)
scored_simulation = await simulate_and_score(simulator, valid_activation_records)
print(f"score={scored_simulation.get_preferred_score():.2f}")

