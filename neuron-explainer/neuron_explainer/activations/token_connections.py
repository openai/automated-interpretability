from dataclasses import dataclass
from typing import List, Union

import blobfile as bf
from neuron_explainer.fast_dataclasses import FastDataclass, loads, register_dataclass


@register_dataclass
@dataclass
class TokensAndWeights(FastDataclass):
    tokens: List[str]
    strengths: List[float]


@register_dataclass
@dataclass
class WeightBasedSummaryOfNeuron(FastDataclass):
    input_positive: TokensAndWeights
    input_negative: TokensAndWeights
    output_positive: TokensAndWeights
    output_negative: TokensAndWeights


def load_token_weight_connections_of_neuron(
    layer_index: Union[str, int], neuron_index: Union[str, int],
    dataset_path: str = "az://openaipublic/neuron-explainer/data/related-tokens/weight-based",
) -> WeightBasedSummaryOfNeuron:
    """Load the TokenLookupTableSummaryOfNeuron for the specified neuron."""
    file = bf.join(dataset_path, f"{layer_index}/{neuron_index}.json")
    with bf.BlobFile(file, "r") as f:
        return loads(f.read(), backwards_compatible=False)


@register_dataclass
@dataclass
class TokenLookupTableSummaryOfNeuron(FastDataclass):
    """List of tokens and the average activations of a given neuron in response to each
    respective token. These are selected from among the tokens in the vocabulary with the
    highest average activations across an internet text dataset, with the highest activations
    first."""

    tokens: List[str]
    average_activations: List[float]


def load_token_lookup_table_connections_of_neuron(
    layer_index: Union[str, int], neuron_index: Union[str, int],
    dataset_path: str = "az://openaipublic/neuron-explainer/data/related-tokens/activation-based",
) -> TokenLookupTableSummaryOfNeuron:
    """Load the TokenLookupTableSummaryOfNeuron for the specified neuron."""
    file = bf.join(dataset_path, f"{layer_index}/{neuron_index}.json")
    with bf.BlobFile(file, "r") as f:
        return loads(f.read(), backwards_compatible=False)

