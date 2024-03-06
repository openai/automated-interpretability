from dataclasses import dataclass
from enum import Enum
from typing import List

from neuron_explainer.fast_dataclasses import FastDataclass


@dataclass
class Example(FastDataclass):
    """
    An example list of tokens as strings corresponding to top token space inputs of a neuron, with a
    string explanation of the neuron's behavior on these tokens.
    """

    tokens: List[str]
    explanation: str


class TokenSpaceFewShotExampleSet(Enum):
    """Determines which few-shot examples to use when sampling explanations."""

    ORIGINAL = "original"
    TEST = "test"

    def get_examples(self) -> list[Example]:
        """Returns regular examples for use in a few-shot prompt."""
        if self is TokenSpaceFewShotExampleSet.ORIGINAL:
            return ORIGINAL_EXAMPLES
        elif self is TokenSpaceFewShotExampleSet.TEST:
            return TEST_EXAMPLES
        else:
            raise ValueError(f"Unhandled example set: {self}")


ORIGINAL_EXAMPLES = [
    Example(
        tokens=[
            "actual",
            " literal",
            " actual",
            " hyper",
            " real",
            " EX",
            " Real",
            "^",
            "Full",
            " full",
            " optical",
            " style",
            "any",
            "ALL",
            "extreme",
            " miniature",
            " Optical",
            " faint",
            "~",
            " Physical",
            " REAL",
            "*",
            "virtual",
            "TYPE",
            " technical",
            "otally",
            " physic",
            "Type",
            "<",
            "images",
            "atic",
            " sheer",
            " Style",
            " partial",
            " natural",
            "Hyper",
            " Any",
            " theoretical",
            "|",
            " ultimate",
            "oing",
            " constant",
            "ANY",
            "antically",
            "ishly",
            " ex",
            " visual",
            "special",
            "omorphic",
            "visual",
        ],
        explanation=" adjectives related to being real, or to physical properties and evidence",
    ),
    Example(
        tokens=[
            "cephal",
            "aeus",
            " coma",
            "bered",
            "abetes",
            "inflamm",
            "rugged",
            "alysed",
            "azine",
            "hered",
            "cells",
            "aneously",
            "fml",
            "igm",
            "culosis",
            "iani",
            "CTV",
            "disabled",
            "heric",
            "ulo",
            "geoning",
            "awi",
            "translation",
            "iral",
            "govtrack",
            "mson",
            "cloth",
            "nesota",
            " Dise",
            " Lyme",
            " dementia",
            "agn",
            " reversible",
            " susceptibility",
            "esthesia",
            "orf",
            " inflamm",
            " Obesity",
            " tox",
            " Disorders",
            "uberty",
            "blind",
            "ALTH",
            "avier",
            " Immunity",
            " Hurt",
            "ulet",
            "ueless",
            " sluggish",
            "rosis",
        ],
        explanation=" words related to physical medical conditions",
    ),
    Example(
        tokens=[
            " January",
            "terday",
            "cember",
            " April",
            " July",
            "September",
            "December",
            "Thursday",
            "quished",
            "November",
            "Tuesday",
            "uesday",
            " Sept",
            "ruary",
            " March",
            ";;;;;;;;;;;;",
            " Monday",
            "Wednesday",
            " Saturday",
            " Wednesday",
            "Reloaded",
            "aturday",
            " August",
            "Feb",
            "Sunday",
            "Reviewed",
            "uggest",
            " Dhabi",
            "ACTED",
            "tten",
            "Year",
            "August",
            "alogue",
            "MX",
            " Janeiro",
            "yss",
            " Leilan",
            " Fiscal",
            " referen",
            "semb",
            "eele",
            "wcs",
            "detail",
            "ertation",
            " Reborn",
            " Sunday",
            "itially",
            "aturdays",
            " Dise",
            "essage",
        ],
        explanation=" nouns related to time and dates",
    ),
]

TEST_EXAMPLES = [
    Example(
        tokens=[
            "these",
            " are",
            " tokens",
        ],
        explanation=" this is a test explanation",
    ),
]
