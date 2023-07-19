# Dataclasses and enums for storing neuron-indexed information about activations. Also, related
# helper functions.

import math
from dataclasses import dataclass, field
from typing import List, Optional, Union

import urllib.request
import blobfile as bf
import boostedblob as bbb
from neuron_explainer.fast_dataclasses import FastDataclass, loads, register_dataclass
from neuron_explainer.azure import standardize_azure_url


@register_dataclass
@dataclass
class ActivationRecord(FastDataclass):
    """Collated lists of tokens and their activations for a single neuron."""

    tokens: List[str]
    """Tokens in the text sequence, represented as strings."""
    activations: List[float]
    """Raw activation values for the neuron on each token in the text sequence."""


@register_dataclass
@dataclass
class NeuronId(FastDataclass):
    """Identifier for a neuron in an artificial neural network."""

    layer_index: int
    """The index of layer the neuron is in. The first layer used during inference has index 0."""
    neuron_index: int
    """The neuron's index within in its layer. Indices start from 0 in each layer."""


def _check_slices(
    slices_by_split: dict[str, slice],
    expected_num_values: int,
) -> None:
    """Assert that the slices are disjoint and fully cover the intended range."""
    indices = set()
    sum_of_slice_lengths = 0
    n_splits = len(slices_by_split.keys())
    for s in slices_by_split.values():
        subrange = range(expected_num_values)[s]
        sum_of_slice_lengths += len(subrange)
        indices |= set(subrange)
    assert (
        sum_of_slice_lengths == expected_num_values
    ), f"{sum_of_slice_lengths=} != {expected_num_values=}"
    stride = n_splits
    expected_indices = set.union(
        *[set(range(start_index, expected_num_values, stride)) for start_index in range(n_splits)]
    )
    assert indices == expected_indices, f"{indices=} != {expected_indices=}"


def get_slices_for_splits(
    splits: list[str],
    num_activation_records_per_split: int,
) -> dict[str, slice]:
    """
    Get equal-sized interleaved subsets for each of a list of splits, given the number of elements
    to include in each split.
    """

    stride = len(splits)
    num_activation_records_for_even_splits = num_activation_records_per_split * stride
    slices_by_split = {
        split: slice(split_index, num_activation_records_for_even_splits, stride)
        for split_index, split in enumerate(splits)
    }
    _check_slices(
        slices_by_split=slices_by_split,
        expected_num_values=num_activation_records_for_even_splits,
    )
    return slices_by_split


@dataclass
class ActivationRecordSliceParams:
    """How to select splits (train, valid, etc.) of activation records."""

    n_examples_per_split: Optional[int]
    """The number of examples to include in each split."""


@register_dataclass
@dataclass
class NeuronRecord(FastDataclass):
    """Neuron-indexed activation data, including summary stats and notable activation records."""

    neuron_id: NeuronId
    """Identifier for the neuron."""

    random_sample: list[ActivationRecord] = field(default_factory=list)
    """
    Random activation records for this neuron. The random sample is independent from those used for
    other neurons.
    """
    random_sample_by_quantile: Optional[list[list[ActivationRecord]]] = None
    """
    Random samples of activation records in each of the specified quantiles. None if quantile
    tracking is disabled.
    """
    quantile_boundaries: Optional[list[float]] = None
    """Boundaries of the quantiles used to generate the random_sample_by_quantile field."""

    # Moments of activations
    mean: Optional[float] = math.nan
    variance: Optional[float] = math.nan
    skewness: Optional[float] = math.nan
    kurtosis: Optional[float] = math.nan

    most_positive_activation_records: list[ActivationRecord] = field(default_factory=list)
    """
    Activation records with the most positive figure of merit value for this neuron over all dataset
    examples.
    """

    @property
    def max_activation(self) -> float:
        """Return the maximum activation value over all top-activating activation records."""
        return max([max(ar.activations) for ar in self.most_positive_activation_records])

    def _get_top_activation_slices(
        self, activation_record_slice_params: ActivationRecordSliceParams
    ) -> dict[str, slice]:
        splits = ["train", "calibration", "valid", "test"]
        n_examples_per_split = activation_record_slice_params.n_examples_per_split
        if n_examples_per_split is None:
            n_examples_per_split = len(self.most_positive_activation_records) // len(splits)
        assert len(self.most_positive_activation_records) >= n_examples_per_split * len(splits)
        return get_slices_for_splits(splits, n_examples_per_split)

    def _get_random_activation_slices(
        self, activation_record_slice_params: ActivationRecordSliceParams
    ) -> dict[str, slice]:
        splits = ["calibration", "valid", "test"]
        n_examples_per_split = activation_record_slice_params.n_examples_per_split
        if n_examples_per_split is None:
            n_examples_per_split = len(self.random_sample) // len(splits)
        # NOTE: this assert could trigger on some old datasets with only 10 random samples, in which case you may have to remove "test" from the set of splits
        assert len(self.random_sample) >= n_examples_per_split * len(splits)
        return get_slices_for_splits(splits, n_examples_per_split)

    def train_activation_records(
        self,
        activation_record_slice_params: ActivationRecordSliceParams,
    ) -> list[ActivationRecord]:
        """
        Train split, typically used for generating explanations. Consists exclusively of
        top-activating records since context window limitations make it difficult to include
        random records.
        """
        return self.most_positive_activation_records[
            self._get_top_activation_slices(activation_record_slice_params)["train"]
        ]

    def calibration_activation_records(
        self,
        activation_record_slice_params: ActivationRecordSliceParams,
    ) -> list[ActivationRecord]:
        """
        Calibration split, typically used for calibrating neuron simulations. See
        http://go/neuron_explanation_methodology for an explanation of calibration. Consists of
        top-activating records and random records in a 1:1 ratio.
        """
        return (
            self.most_positive_activation_records[
                self._get_top_activation_slices(activation_record_slice_params)["calibration"]
            ]
            + self.random_sample[
                self._get_random_activation_slices(activation_record_slice_params)["calibration"]
            ]
        )

    def valid_activation_records(
        self,
        activation_record_slice_params: ActivationRecordSliceParams,
    ) -> list[ActivationRecord]:
        """
        Validation split, typically used for evaluating explanations, either automatically with
        simulation + correlation coefficient scoring, or manually by humans. Consists of
        top-activating records and random records in a 1:1 ratio.
        """
        return (
            self.most_positive_activation_records[
                self._get_top_activation_slices(activation_record_slice_params)["valid"]
            ]
            + self.random_sample[
                self._get_random_activation_slices(activation_record_slice_params)["valid"]
            ]
        )

    def test_activation_records(
        self,
        activation_record_slice_params: ActivationRecordSliceParams,
    ) -> list[ActivationRecord]:
        """
        Test split, typically used for explanation evaluations that can't use the validation split.
        Consists of top-activating records and random records in a 1:1 ratio.
        """
        return (
            self.most_positive_activation_records[
                self._get_top_activation_slices(activation_record_slice_params)["test"]
            ]
            + self.random_sample[
                self._get_random_activation_slices(activation_record_slice_params)["test"]
            ]
        )


def neuron_exists(
    dataset_path: str, layer_index: Union[str, int], neuron_index: Union[str, int]
) -> bool:
    """Return whether the specified neuron exists."""
    file = bf.join(dataset_path, "neurons", str(layer_index), f"{neuron_index}.json")
    return bf.exists(file)


def load_neuron(
    layer_index: Union[str, int],
    neuron_index: Union[str, int],
    dataset_path: str = "https://openaipublic.blob.core.windows.net/neuron-explainer/data/collated-activations",
) -> NeuronRecord:
    """Load the NeuronRecord for the specified neuron."""
    url = "/".join([dataset_path, str(layer_index), f"{neuron_index}.json"])
    url = standardize_azure_url(url)
    with urllib.request.urlopen(url) as f:
        neuron_record = loads(f.read())
        if not isinstance(neuron_record, NeuronRecord):
            raise ValueError(
                f"Stored data incompatible with current version of NeuronRecord dataclass."
            )
        return neuron_record


@bbb.ensure_session
async def load_neuron_async(
    layer_index: Union[str, int],
    neuron_index: Union[str, int],
    dataset_path: str = "az://openaipublic/neuron-explainer/data/collated-activations",
) -> NeuronRecord:
    """Async version of load_neuron."""
    file = bf.join(dataset_path, str(layer_index), f"{neuron_index}.json")
    return await read_neuron_file(file)


@bbb.ensure_session
async def read_neuron_file(neuron_filename: str) -> NeuronRecord:
    """Like load_neuron_async, but takes a raw neuron filename."""
    raw_contents = await bbb.read.read_single(neuron_filename)
    neuron_record = loads(raw_contents.decode("utf-8"))
    if not isinstance(neuron_record, NeuronRecord):
        raise ValueError(
            f"Stored data incompatible with current version of NeuronRecord dataclass."
        )
    return neuron_record


def get_sorted_neuron_indices(dataset_path: str, layer_index: Union[str, int]) -> List[int]:
    """Returns the indices of all neurons in this layer, in ascending order."""
    layer_dir = bf.join(dataset_path, "neurons", str(layer_index))
    return sorted(
        [int(f.split(".")[0]) for f in bf.listdir(layer_dir) if f.split(".")[0].isnumeric()]
    )


def get_sorted_layers(dataset_path: str) -> List[str]:
    """
    Return the indices of all layers in this dataset, in ascending numerical order, as strings.
    """
    return [
        str(x)
        for x in sorted(
            [int(x) for x in bf.listdir(bf.join(dataset_path, "neurons")) if x.isnumeric()]
        )
    ]
