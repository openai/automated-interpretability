# Dataclasses and enums for storing neuron explanations, their scores, and related data. Also,
# related helper functions.

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import blobfile as bf
import boostedblob as bbb
from neuron_explainer.activations.activations import NeuronId
from neuron_explainer.fast_dataclasses import FastDataclass, loads, register_dataclass


class ActivationScale(str, Enum):
    """Which "units" are stored in the expected_activations/distribution_values fields of a
    SequenceSimulation.

    This enum identifies whether the values represent real activations of the neuron or something
    else. Different scales are not necessarily related by a linear transformation.
    """

    NEURON_ACTIVATIONS = "neuron_activations"
    """Values represent real activations of the neuron."""
    SIMULATED_NORMALIZED_ACTIVATIONS = "simulated_normalized_activations"
    """
    Values represent simulated activations of the neuron, normalized to the range [0, 10]. This
    scale is arbitrary and should not be interpreted as a neuron activation.
    """


@register_dataclass
@dataclass
class SequenceSimulation(FastDataclass):
    """The result of a simulation of neuron activations on one text sequence."""

    tokens: list[str]
    """The sequence of tokens that was simulated."""
    expected_activations: list[float]
    """Expected value of the possibly-normalized activation for each token in the sequence."""
    activation_scale: ActivationScale
    """What scale is used for values in the expected_activations field."""
    distribution_values: list[list[float]]
    """
    For each token in the sequence, a list of values from the discrete distribution of activations
    produced from simulation. Tokens will be included here if and only if they are in the top K=15
    tokens predicted by the simulator, and excluded otherwise.
    
    May be transformed to another unit by calibration. When we simulate a neuron, we produce a
    discrete distribution with values in the arbitrary discretized space of the neuron, e.g. 10%
    chance of 0, 70% chance of 1, 20% chance of 2. Which we store as distribution_values =
    [0, 1, 2], distribution_probabilities = [0.1, 0.7, 0.2]. When we transform the distribution to
    the real activation units, we can correspondingly transform the values of this distribution
    to get a distribution in the units of the neuron. e.g. if the mapping from the discretized space
    to the real activation unit of the neuron is f(x) = x/2, then the distribution becomes 10%
    chance of 0, 70% chance of 0.5, 20% chance of 1. Which we store as distribution_values =
    [0, 0.5, 1], distribution_probabilities = [0.1, 0.7, 0.2].
    """
    distribution_probabilities: list[list[float]]
    """
    For each token in the sequence, the probability of the corresponding value in
    distribution_values.
    """

    uncalibrated_simulation: Optional["SequenceSimulation"] = None
    """The result of the simulation before calibration."""


@register_dataclass
@dataclass
class ScoredSequenceSimulation(FastDataclass):
    """
    SequenceSimulation result with a score (for that sequence only) and ground truth activations.
    """

    simulation: SequenceSimulation
    """The result of a simulation of neuron activations."""
    true_activations: List[float]
    """Ground truth activations on the sequence (not normalized)"""
    ev_correlation_score: float
    """
    Correlation coefficient between the expected values of the normalized activations from the
    simulation and the unnormalized true activations of the neuron on the text sequence.
    """
    rsquared_score: Optional[float] = None
    """R^2 of the simulated activations."""
    absolute_dev_explained_score: Optional[float] = None
    """
    Score based on absolute difference between real and simulated activations.
    absolute_dev_explained_score = 1 - mean(abs(real-predicted))/ mean(abs(real))
    """


@register_dataclass
@dataclass
class ScoredSimulation(FastDataclass):
    """Result of scoring a neuron simulation on multiple sequences."""

    scored_sequence_simulations: List[ScoredSequenceSimulation]
    """ScoredSequenceSimulation for each sequence"""
    ev_correlation_score: Optional[float] = None
    """
    Correlation coefficient between the expected values of the normalized activations from the
    simulation and the unnormalized true activations on a dataset created from all score_results.
    (Note that this is not equivalent to averaging across sequences.)
    """
    rsquared_score: Optional[float] = None
    """R^2 of the simulated activations."""
    absolute_dev_explained_score: Optional[float] = None
    """
    Score based on absolute difference between real and simulated activations.
    absolute_dev_explained_score = 1 - mean(abs(real-predicted))/ mean(abs(real)).
    """

    def get_preferred_score(self) -> Optional[float]:
        """
        This method may return None in cases where the score is undefined, for example if the
        normalized activations were all zero, yielding a correlation coefficient of NaN.
        """
        return self.ev_correlation_score


@register_dataclass
@dataclass
class ScoredExplanation(FastDataclass):
    """Simulator parameters and the results of scoring it on multiple sequences"""

    explanation: str
    """The explanation used for simulation."""

    scored_simulation: ScoredSimulation
    """Result of scoring the neuron simulator on multiple sequences."""

    def get_preferred_score(self) -> Optional[float]:
        """
        This method may return None in cases where the score is undefined, for example if the
        normalized activations were all zero, yielding a correlation coefficient of NaN.
        """
        return self.scored_simulation.get_preferred_score()


@register_dataclass
@dataclass
class NeuronSimulationResults(FastDataclass):
    """Simulation results and scores for a neuron."""

    neuron_id: NeuronId
    scored_explanations: list[ScoredExplanation]


def load_neuron_explanations(
    explanations_path: str, layer_index: Union[str, int], neuron_index: Union[str, int]
) -> Optional[NeuronSimulationResults]:
    """Load scored explanations for the specified neuron."""
    file = bf.join(explanations_path, str(layer_index), f"{neuron_index}.jsonl")
    if not bf.exists(file):
        return None
    with bf.BlobFile(file) as f:
        for line in f:
            return loads(line)
    return None


@bbb.ensure_session
async def load_neuron_explanations_async(
    explanations_path: str, layer_index: Union[str, int], neuron_index: Union[str, int]
) -> Optional[NeuronSimulationResults]:
    """Load scored explanations for the specified neuron, asynchronously."""
    return await read_explanation_file(
        bf.join(explanations_path, str(layer_index), f"{neuron_index}.jsonl")
    )


@bbb.ensure_session
async def read_file(filename: str) -> Optional[str]:
    """Read the contents of the given file as a string, asynchronously."""
    try:
        raw_contents = await bbb.read.read_single(filename)
    except FileNotFoundError:
        print(f"Could not read {filename}")
        return None
    lines = []
    for line in raw_contents.decode("utf-8").split("\n"):
        if len(line) > 0:
            lines.append(line)
    assert len(lines) == 1, filename
    return lines[0]


@bbb.ensure_session
async def read_explanation_file(explanation_filename: str) -> Optional[NeuronSimulationResults]:
    """Load scored explanations from the given filename, asynchronously."""
    line = await read_file(explanation_filename)
    return loads(line) if line is not None else None


@bbb.ensure_session
async def read_json_file(filename: str) -> Optional[dict]:
    """Read the contents of the given file as a JSON object, asynchronously."""
    line = await read_file(filename)
    return json.loads(line) if line is not None else None


def get_numerical_subdirs(dataset_path: str) -> list[str]:
    """Return the names of all numbered subdirectories in the specified directory.

    Used to get all layer directories in an explanation directory.
    """
    return [
        str(x)
        for x in sorted(
            [
                int(x)
                for x in bf.listdir(dataset_path)
                if bf.isdir(bf.join(dataset_path, x)) and x.isnumeric()
            ]
        )
    ]


def get_sorted_neuron_indices_from_explanations(
    explanations_path: str, layer: Union[str, int]
) -> list[int]:
    """Return the indices of all neurons in this layer, in ascending order."""
    layer_dir = bf.join(explanations_path, str(layer))
    return sorted(
        [int(f.split(".")[0]) for f in bf.listdir(layer_dir) if f.split(".")[0].isnumeric()]
    )
