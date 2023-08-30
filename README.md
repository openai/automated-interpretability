# Automated interpretability

## Code and tools

This repository contains code and tools associated with the [Language models can explain neurons in
language models](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) paper, specifically:

* Code for automatically generating, simulating, and scoring explanations of neuron behavior using
the methodology described in the paper. See the
[neuron-explainer README](neuron-explainer/README.md) for more information.

Note: if you run into errors of the form "Error: Could not find any credentials that grant access to storage account: 'openaipublic' and container: 'neuron-explainer'"." you might be able to fix this by signing up for an azure account and specifying the credentials as described in the error message. 

* A tool for viewing neuron activations and explanations, accessible
[here](https://openaipublic.blob.core.windows.net/neuron-explainer/neuron-viewer/index.html). See
the [neuron-viewer README](neuron-viewer/README.md) for more information.

## Public datasets

Together with this code, we're also releasing public datasets of GPT-2 XL neurons and explanations.
Here's an overview of those datasets.  

* Neuron activations: `az://openaipublic/neuron-explainer/data/collated-activations/{layer_index}/{neuron_index}.json`
    - Tokenized text sequences and their activations for the neuron. We
    provide multiple sets of tokens and activations: top-activating ones, random
    samples from several quantiles; and a completely random sample. We also provide
    some basic statistics for the activations.
    - Each file contains a JSON-formatted
    [`NeuronRecord`](neuron-explainer/neuron_explainer/activations/activations.py#L89) dataclass.
* Neuron explanations: `az://openaipublic/neuron-explainer/data/explanations/{layer_index}/{neuron_index}.jsonl`
    - Scored model-generated explanations of the behavior of the neuron, including simulation results.
    - Each file contains a JSON-formatted
    [`NeuronSimulationResults`](neuron-explainer/neuron_explainer/explanations/explanations.py#L146)
    dataclass.
* Related neurons: `az://openaipublic/neuron-explainer/data/related-neurons/weight-based/{layer_index}/{neuron_index}.json`
    - Lists of the upstream and downstream neurons with the most positive and negative connections (see below for definition).
    - Each file contains a JSON-formatted dataclass whose definition is not included in this repo.
* Tokens with high average activations:
`az://openaipublic/neuron-explainer/data/related-tokens/activation-based/{layer_index}/{neuron_index}.json`
    - Lists of tokens with the highest average activations for individual neurons, and their average activations.
    - Each file contains a JSON-formatted [`TokenLookupTableSummaryOfNeuron`](neuron-explainer/neuron_explainer/activations/token_connections.py#L36)
    dataclass.
* Tokens with large inbound and outbound weights:
`az://openaipublic/neuron-explainer/data/related-tokens/weight-based/{layer_index}/{neuron_index}.json`
    - List of the most-positive and most-negative input and output tokens for individual neurons,
    as well as the associated weight (see below for definition). 
    - Each file contains a JSON-formatted [`WeightBasedSummaryOfNeuron`](neuron-explainer/neuron_explainer/activations/token_connections.py#L17)
    dataclass.

Update (July 5, 2023):
We also released a set of explanations for GPT-2 Small. The methodology is slightly different from the methodology used for GPT-2 XL so the results aren't directly comparable.
* Neuron activations: `az://openaipublic/neuron-explainer/gpt2_small_data/collated-activations/{layer_index}/{neuron_index}.json`
* Neuron explanations: `az://openaipublic/neuron-explainer/gpt2_small_data/explanations/{layer_index}/{neuron_index}.jsonl`

Update (August 30, 2023): We recently discovered a bug in how we performed inference on the GPT-2 series models used for the paper and for these datasets. Specifically, we used an optimized GELU implementation rather than the original GELU implementation associated with GPT-2. While the model’s behavior is very similar across these two configurations, the post-MLP activation values we used to generate and simulate explanations differ from the correct values by the following amounts for GPT-2 small:

- Median: 0.0090
- 90th percentile: 0.0252
- 99th percentile: 0.0839
- 99.9th percentile: 0.1736

### Definition of connection weights

Refer to [GPT-2 model code](https://github.com/openai/gpt-2/blob/master/src/model.py) for
understanding of model weight conventions.

*Neuron-neuron*: For two neurons `(l1, n1)` and `(l2, n2)` with `l1 < l2`, the connection strength is defined as
`h{l1}.mlp.c_proj.w[:, n1, :] @ diag(h{l2}.ln_2.g) @ h{l2}.mlp.c_fc.w[:, :, n2]`.

*Neuron-token*: For token `t` and neuron `(l, n)`, the input weight is computed as
`wte[t, :] @ diag(h{l}.ln_2.g) @ h{l}.mlp.c_fc.w[:, :, n]`
and the output weight is computed as
`h{l}.mlp.c_proj.w[:, n, :] @ diag(ln_f.g) @ wte[t, :]`.

### Misc Lists of Interesting Neurons
Lists of neurons we thought were interesting according to different criteria, with some preliminary descriptions.
* [Interesting Neurons (external)](https://docs.google.com/spreadsheets/d/1p7fYs31NU8sJoeKyUx4Mn2laGx8xXfHg_KcIvYiKPpg/edit#gid=0)
* [Neurons that score high on random, possibly monosemantic? (external)](https://docs.google.com/spreadsheets/d/1TqKFcz-84jyIHLU7VRoTc8BoFBMpbgac-iNBnxVurQ8/edit?usp=sharing)
* [Clusters of neurons well explained by activation explanation but not by tokens](https://docs.google.com/document/d/1lWhKowpKDdwTMALD_K541cdwgGoQx8DFUSuEe1U2AGE/edit?usp=sharing)
* [Neurons sensitive to truncation](https://docs.google.com/document/d/1x89TWBvuHcyC2t01EDbJZJ5LQYHozlcS-VUmr5shf_A/edit?usp=sharing)
