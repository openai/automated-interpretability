import json
import os
from dataclasses import dataclass

from neuron_explainer.activations.activations import ActivationRecord


@dataclass(frozen=True)
class Puzzle:
    """A puzzle is a ground truth explanation, a collection of sentences (stored as ActivationRecords) with activations
    according to that explanation, and a collection of false explanations"""

    name: str
    explanation: str
    activation_records: list[ActivationRecord]
    false_explanations: list[str]


def convert_puzzle_to_tokenized_sentences(puzzle: Puzzle) -> list[list[str]]:
    """Converts a puzzle to a list of tokenized sentences."""
    return [record.tokens for record in puzzle.activation_records]


def convert_puzzle_dict_to_puzzle(puzzle_dict: dict) -> Puzzle:
    """Converts a json dictionary representation of a puzzle to the Puzzle class."""
    puzzle_activation_records = []
    for sentence in puzzle_dict["sentences"]:
        # Token-activation pairs are listed as either a string or a list of a string and a float. If it is a list, the float is the activation.
        # If it is only a string, the activation is assumed to be 0. This is useful for readability and reducing redundancy in the data.
        tokens = [t[0] if type(t) is list else t for t in sentence]
        assert all([type(t) is str for t in tokens]), "All tokens must be strings"
        activations = [float(t[1]) if type(t) is list else 0.0 for t in sentence]
        assert all([type(t) is float for t in activations]), "All activations must be floats"

        puzzle_activation_records.append(ActivationRecord(tokens=tokens, activations=activations))

    return Puzzle(
        name=puzzle_dict["name"],
        explanation=puzzle_dict["explanation"],
        activation_records=puzzle_activation_records,
        false_explanations=puzzle_dict["false_explanations"],
    )


PUZZLES_BY_NAME: dict[str, Puzzle] = dict()
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "puzzles.json"), "r") as f:
    puzzle_dicts = json.loads(f.read())
    for name in puzzle_dicts.keys():
        PUZZLES_BY_NAME[name] = convert_puzzle_dict_to_puzzle(puzzle_dicts[name])
