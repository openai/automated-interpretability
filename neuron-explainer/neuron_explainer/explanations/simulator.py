"""Uses API calls to simulate neuron activations based on an explanation."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import Any, Optional, Sequence, Union

import numpy as np
from neuron_explainer.activations.activation_records import (
    calculate_max_activation,
    format_activation_records,
    format_sequences_for_simulation,
    normalize_activations,
)
from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.api_client import ApiClient
from neuron_explainer.explanations.explainer import EXPLANATION_PREFIX
from neuron_explainer.explanations.explanations import ActivationScale, SequenceSimulation
from neuron_explainer.explanations.few_shot_examples import FewShotExampleSet
from neuron_explainer.explanations.prompt_builder import (
    HarmonyMessage,
    PromptBuilder,
    PromptFormat,
    Role,
)

logger = logging.getLogger(__name__)

# Our prompts use normalized activation values, which map any range of positive activations to the
# integers from 0 to 10.
MAX_NORMALIZED_ACTIVATION = 10
VALID_ACTIVATION_TOKENS_ORDERED = list(str(i) for i in range(MAX_NORMALIZED_ACTIVATION + 1))
VALID_ACTIVATION_TOKENS = set(VALID_ACTIVATION_TOKENS_ORDERED)


class SimulationType(str, Enum):
    """How to simulate neuron activations. Values correspond to subclasses of NeuronSimulator."""

    ALL_AT_ONCE = "all_at_once"
    """
    Use a single prompt with <unknown> tokens; calculate EVs using logprobs.
    
    Implemented by ExplanationNeuronSimulator.
    """

    ONE_AT_A_TIME = "one_at_a_time"
    """
    Use a separate prompt for each token being simulated; calculate EVs using logprobs.
    
    Implemented by ExplanationTokenByTokenSimulator.
    """

    @classmethod
    def from_string(cls, s: str) -> SimulationType:
        for simulation_type in SimulationType:
            if simulation_type.value == s:
                return simulation_type
        raise ValueError(f"Invalid simulation type: {s}")


def compute_expected_value(
    norm_probabilities_by_distribution_value: OrderedDict[int, float]
) -> float:
    """
    Given a map from distribution values (integers on the range [0, 10]) to normalized
    probabilities, return an expected value for the distribution.
    """
    return np.dot(
        np.array(list(norm_probabilities_by_distribution_value.keys())),
        np.array(list(norm_probabilities_by_distribution_value.values())),
    )


def parse_top_logprobs(top_logprobs: dict[str, float]) -> OrderedDict[int, float]:
    """
    Given a map from tokens to logprobs, return a map from distribution values (integers on the
    range [0, 10]) to unnormalized probabilities (in the sense that they may not sum to 1).
    """
    probabilities_by_distribution_value = OrderedDict()
    for token, logprob in top_logprobs.items():
        if token in VALID_ACTIVATION_TOKENS:
            token_as_int = int(token)
            probabilities_by_distribution_value[token_as_int] = np.exp(logprob)
    return probabilities_by_distribution_value


def compute_predicted_activation_stats_for_token(
    top_logprobs: dict[str, float],
) -> tuple[OrderedDict[int, float], float]:
    probabilities_by_distribution_value = parse_top_logprobs(top_logprobs)
    total_p_of_distribution_values = sum(probabilities_by_distribution_value.values())
    norm_probabilities_by_distribution_value = OrderedDict(
        {
            distribution_value: p / total_p_of_distribution_values
            for distribution_value, p in probabilities_by_distribution_value.items()
        }
    )
    expected_value = compute_expected_value(norm_probabilities_by_distribution_value)
    return (
        norm_probabilities_by_distribution_value,
        expected_value,
    )


# Adapted from tether/tether/core/encoder.py.
def convert_to_byte_array(s: str) -> bytearray:
    byte_array = bytearray()
    assert s.startswith("bytes:"), s
    s = s[6:]
    while len(s) > 0:
        if s[0] == "\\":
            # Hex encoding.
            assert s[1] == "x"
            assert len(s) >= 4
            byte_array.append(int(s[2:4], 16))
            s = s[4:]
        else:
            # Regular ascii encoding.
            byte_array.append(ord(s[0]))
            s = s[1:]
    return byte_array


def handle_byte_encoding(
    response_tokens: Sequence[str], merged_response_index: int
) -> tuple[str, int]:
    """
    Handle the case where the current token is a sequence of bytes. This may involve merging
    multiple response tokens into a single token.
    """
    response_token = response_tokens[merged_response_index]
    if response_token.startswith("bytes:"):
        byte_array = bytearray()
        while True:
            byte_array = convert_to_byte_array(response_token) + byte_array
            try:
                # If we can decode the byte array as utf-8, then we're done.
                response_token = byte_array.decode("utf-8")
                break
            except UnicodeDecodeError:
                # If not, then we need to merge the previous response token into the byte
                # array.
                merged_response_index -= 1
                response_token = response_tokens[merged_response_index]
    return response_token, merged_response_index


def was_token_split(current_token: str, response_tokens: Sequence[str], start_index: int) -> bool:
    """
    Return whether current_token (a token from the subject model) was split into multiple tokens by
    the simulator model (as represented by the tokens in response_tokens). start_index is the index
    in response_tokens at which to begin looking backward to form a complete token. It is usually
    the first token *before* the delimiter that separates the token from the normalized activation,
    barring some unusual cases.

    This mainly happens if the subject model uses a different tokenizer than the simulator model.
    But it can also happen in cases where Unicode characters are split. This function handles both
    cases.
    """
    merged_response_tokens = ""
    merged_response_index = start_index
    while len(merged_response_tokens) < len(current_token):
        response_token = response_tokens[merged_response_index]
        response_token, merged_response_index = handle_byte_encoding(
            response_tokens, merged_response_index
        )
        merged_response_tokens = response_token + merged_response_tokens
        merged_response_index -= 1
    # It's possible that merged_response_tokens is longer than current_token at this point,
    # since the between-lines delimiter may have been merged into the original token. But it
    # should always be the case that merged_response_tokens ends with current_token.
    assert merged_response_tokens.endswith(current_token)
    num_merged_tokens = start_index - merged_response_index
    token_was_split = num_merged_tokens > 1
    if token_was_split:
        logger.debug(
            "Warning: token from the subject model was split into 2+ tokens by the simulator model."
        )
    return token_was_split


def parse_simulation_response(
    response: dict[str, Any],
    prompt_format: PromptFormat,
    tokens: Sequence[str],
) -> SequenceSimulation:
    """
    Parse an API response to a simulation prompt.

    Args:
        response: response from the API
        prompt_format: how the prompt was formatted
        tokens: list of tokens as strings in the sequence where the neuron is being simulated
    """
    choice = response["choices"][0]
    if prompt_format == PromptFormat.HARMONY_V4:
        text = choice["message"]["content"]
    elif prompt_format in [
        PromptFormat.NONE,
        PromptFormat.INSTRUCTION_FOLLOWING,
    ]:
        text = choice["text"]
    else:
        raise ValueError(f"Unhandled prompt format {prompt_format}")
    response_tokens = choice["logprobs"]["tokens"]
    choice["logprobs"]["token_logprobs"]
    top_logprobs = choice["logprobs"]["top_logprobs"]
    token_text_offset = choice["logprobs"]["text_offset"]
    # This only works because the sequence "<start>" tokenizes into multiple tokens if it appears in
    # a text sequence in the prompt.
    scoring_start = text.rfind("<start>")
    expected_values = []
    original_sequence_tokens: list[str] = []
    distribution_values: list[list[float]] = []
    distribution_probabilities: list[list[float]] = []
    for i in range(2, len(response_tokens)):
        if len(original_sequence_tokens) == len(tokens):
            # Make sure we haven't hit some sort of off-by-one error.
            # TODO(sbills): Generalize this to handle different tokenizers.
            reached_end = response_tokens[i + 1] == "<" and response_tokens[i + 2] == "end"
            assert reached_end, f"{response_tokens[i-3:i+3]}"
            break
        if token_text_offset[i] >= scoring_start:
            # We're looking for the first token after a tab. This token should be the text
            # "unknown" if hide_activations=True or a normalized activation (0-10) otherwise.
            # If it isn't, that means that the tab is not appearing as a delimiter, but rather
            # as a token, in which case we should move on to the next response token.
            if response_tokens[i - 1] == "\t":
                if response_tokens[i] != "unknown":
                    logger.debug("Ignoring tab token that is not followed by an 'unknown' token.")
                    continue

                # j represents the index of the token in a "token<tab>activation" line, barring
                # one of the unusual cases handled below.
                j = i - 2

                current_token = tokens[len(original_sequence_tokens)]
                if current_token == response_tokens[j] or was_token_split(
                    current_token, response_tokens, j
                ):
                    # We're in the normal case where the tokenization didn't throw off the
                    # formatting or in the token-was-split case, which we handle the usual way.
                    current_top_logprobs = top_logprobs[i]

                    (
                        norm_probabilities_by_distribution_value,
                        expected_value,
                    ) = compute_predicted_activation_stats_for_token(
                        current_top_logprobs,
                    )
                    current_distribution_values = list(
                        norm_probabilities_by_distribution_value.keys()
                    )
                    current_distribution_probabilities = list(
                        norm_probabilities_by_distribution_value.values()
                    )
                else:
                    # We're in a case where the tokenization resulted in a newline being folded into
                    # the token. We can't do our usual prediction of activation stats for the token,
                    # since the model did not observe the original token. Instead, we use dummy
                    # values. See the TODO elsewhere in this file about coming up with a better
                    # prompt format that avoids this situation.
                    newline_folded_into_token = "\n" in response_tokens[j]
                    assert (
                        newline_folded_into_token
                    ), f"`{current_token=}` {response_tokens[j-3:j+3]=}"
                    logger.debug(
                        "Warning: newline before a token<tab>activation line was folded into the token"
                    )
                    current_distribution_values = []
                    current_distribution_probabilities = []
                    expected_value = 0.0

                original_sequence_tokens.append(current_token)
                distribution_values.append([float(v) for v in current_distribution_values])
                distribution_probabilities.append(current_distribution_probabilities)
                expected_values.append(expected_value)

    return SequenceSimulation(
        tokens=original_sequence_tokens,
        expected_activations=expected_values,
        activation_scale=ActivationScale.SIMULATED_NORMALIZED_ACTIVATIONS,
        distribution_values=distribution_values,
        distribution_probabilities=distribution_probabilities,
    )


class NeuronSimulator(ABC):
    """Abstract base class for simulating neuron behavior."""

    @abstractmethod
    async def simulate(self, tokens: Sequence[str]) -> SequenceSimulation:
        """Simulate the behavior of a neuron based on an explanation."""
        ...


class ExplanationNeuronSimulator(NeuronSimulator):
    """
    Simulate neuron behavior based on an explanation.

    This class uses a few-shot prompt with examples of other explanations and activations. This
    prompt allows us to score all of the tokens at once using a nifty trick involving logprobs.
    """

    def __init__(
        self,
        model_name: str,
        explanation: str,
        max_concurrent: Optional[int] = 10,
        few_shot_example_set: FewShotExampleSet = FewShotExampleSet.ORIGINAL,
        prompt_format: PromptFormat = PromptFormat.INSTRUCTION_FOLLOWING,
        cache: bool = False,
    ):
        self.api_client = ApiClient(
            model_name=model_name, max_concurrent=max_concurrent, cache=cache
        )
        self.explanation = explanation
        self.few_shot_example_set = few_shot_example_set
        self.prompt_format = prompt_format

    async def simulate(
        self,
        tokens: Sequence[str],
    ) -> SequenceSimulation:
        prompt = self.make_simulation_prompt(tokens)

        generate_kwargs: dict[str, Any] = {
            "max_tokens": 0,
            "echo": True,
            "logprobs": 15,
        }
        if self.prompt_format == PromptFormat.HARMONY_V4:
            assert isinstance(prompt, list)
            assert isinstance(prompt[0], dict)  # Really a HarmonyMessage
            generate_kwargs["messages"] = prompt
        else:
            assert isinstance(prompt, str)
            generate_kwargs["prompt"] = prompt

        response = await self.api_client.make_request(**generate_kwargs)
        logger.debug("response in score_explanation_by_activations is %s", response)
        result = parse_simulation_response(response, self.prompt_format, tokens)
        logger.debug("result in score_explanation_by_activations is %s", result)
        return result

    # TODO(sbills): The current token<tab>activation format can result in improper tokenization.
    # In particular, if the token is itself a tab, we may get a single "\t\t" token rather than two
    # "\t" tokens. Consider using a separator that does not appear in any multi-character tokens.
    def make_simulation_prompt(self, tokens: Sequence[str]) -> Union[str, list[HarmonyMessage]]:
        """Create a few-shot prompt for predicting neuron activations for the given tokens."""

        # TODO(sbills): The prompts in this file are subtly different from the ones in explainer.py.
        # Consider reconciling them.
        prompt_builder = PromptBuilder()
        prompt_builder.add_message(
            Role.SYSTEM,
            """We're studying neurons in a neural network.
Each neuron looks for some particular thing in a short document.
Look at summary of what the neuron does, and try to predict how it will fire on each token.

The activation format is token<tab>activation, activations go from 0 to 10, "unknown" indicates an unknown activation. Most activations will be 0.
""",
        )

        few_shot_examples = self.few_shot_example_set.get_examples()
        for i, example in enumerate(few_shot_examples):
            prompt_builder.add_message(
                Role.USER,
                f"\n\nNeuron {i + 1}\nExplanation of neuron {i + 1} behavior: {EXPLANATION_PREFIX} "
                f"{example.explanation}",
            )
            formatted_activation_records = format_activation_records(
                example.activation_records,
                calculate_max_activation(example.activation_records),
                start_indices=example.first_revealed_activation_indices,
            )
            prompt_builder.add_message(
                Role.ASSISTANT, f"\nActivations: {formatted_activation_records}\n"
            )

        prompt_builder.add_message(
            Role.USER,
            f"\n\nNeuron {len(few_shot_examples) + 1}\nExplanation of neuron "
            f"{len(few_shot_examples) + 1} behavior: {EXPLANATION_PREFIX} "
            f"{self.explanation.strip()}",
        )
        prompt_builder.add_message(
            Role.ASSISTANT, f"\nActivations: {format_sequences_for_simulation([tokens])}"
        )
        return prompt_builder.build(self.prompt_format)


class ExplanationTokenByTokenSimulator(NeuronSimulator):
    """
    Simulate neuron behavior based on an explanation.

    Unlike ExplanationNeuronSimulator, this class uses one few-shot prompt per token to calculate
    expected activations. This is slower. This class gets a one-token completion and calculates an
    expected value from that token's logprobs.
    """

    def __init__(
        self,
        model_name: str,
        explanation: str,
        max_concurrent: Optional[int] = 10,
        few_shot_example_set: FewShotExampleSet = FewShotExampleSet.NEWER,
        prompt_format: PromptFormat = PromptFormat.INSTRUCTION_FOLLOWING,
        cache: bool = False,
    ):
        assert (
            few_shot_example_set != FewShotExampleSet.ORIGINAL
        ), "This simulator doesn't support the ORIGINAL few-shot example set."
        self.api_client = ApiClient(
            model_name=model_name, max_concurrent=max_concurrent, cache=cache
        )
        self.explanation = explanation
        self.few_shot_example_set = few_shot_example_set
        self.prompt_format = prompt_format

    async def simulate(
        self,
        tokens: Sequence[str],
    ) -> SequenceSimulation:
        responses_by_token = await asyncio.gather(
            *[
                self._get_activation_stats_for_single_token(tokens, self.explanation, token_index)
                for token_index in range(len(tokens))
            ]
        )
        expected_values, distribution_values, distribution_probabilities = [], [], []
        for response in responses_by_token:
            activation_logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]
            (
                norm_probabilities_by_distribution_value,
                expected_value,
            ) = compute_predicted_activation_stats_for_token(
                activation_logprobs,
            )
            distribution_values.append(
                [float(v) for v in norm_probabilities_by_distribution_value.keys()]
            )
            distribution_probabilities.append(
                list(norm_probabilities_by_distribution_value.values())
            )
            expected_values.append(expected_value)

        result = SequenceSimulation(
            tokens=list(tokens),  # SequenceSimulation expects List type
            expected_activations=expected_values,
            activation_scale=ActivationScale.SIMULATED_NORMALIZED_ACTIVATIONS,
            distribution_values=distribution_values,
            distribution_probabilities=distribution_probabilities,
        )
        logger.debug("result in score_explanation_by_activations is %s", result)
        return result

    async def _get_activation_stats_for_single_token(
        self,
        tokens: Sequence[str],
        explanation: str,
        token_index_to_score: int,
    ) -> dict:
        prompt = self.make_single_token_simulation_prompt(
            tokens,
            explanation,
            token_index_to_score=token_index_to_score,
        )
        return await self.api_client.make_request(
            prompt=prompt, max_tokens=1, echo=False, logprobs=15
        )

    def _add_single_token_simulation_subprompt(
        self,
        prompt_builder: PromptBuilder,
        activation_record: ActivationRecord,
        neuron_index: int,
        explanation: str,
        token_index_to_score: int,
        end_of_prompt: bool,
    ) -> None:
        trimmed_activation_record = ActivationRecord(
            tokens=activation_record.tokens[: token_index_to_score + 1],
            activations=activation_record.activations[: token_index_to_score + 1],
        )
        prompt_builder.add_message(
            Role.USER,
            f"""
Neuron {neuron_index}
Explanation of neuron {neuron_index} behavior: {EXPLANATION_PREFIX} {explanation.strip()}
Text:
{"".join(trimmed_activation_record.tokens)}

Last token in the text:
{trimmed_activation_record.tokens[-1]}

Last token activation, considering the token in the context in which it appeared in the text:
""",
        )
        if not end_of_prompt:
            normalized_activations = normalize_activations(
                trimmed_activation_record.activations, calculate_max_activation([activation_record])
            )
            prompt_builder.add_message(
                Role.ASSISTANT, str(normalized_activations[-1]) + ("" if end_of_prompt else "\n\n")
            )

    def make_single_token_simulation_prompt(
        self,
        tokens: Sequence[str],
        explanation: str,
        token_index_to_score: int,
    ) -> Union[str, list[HarmonyMessage]]:
        """Make a few-shot prompt for predicting the neuron's activation on a single token."""
        assert explanation != ""
        prompt_builder = PromptBuilder()
        prompt_builder.add_message(
            Role.SYSTEM,
            """We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at  an explanation of what the neuron does, and try to predict its activations on a particular token.

The activation format is token<tab>activation, and activations range from 0 to 10. Most activations will be 0.

""",
        )

        few_shot_examples = self.few_shot_example_set.get_examples()
        for i, example in enumerate(few_shot_examples):
            prompt_builder.add_message(
                Role.USER,
                f"Neuron {i + 1}\nExplanation of neuron {i + 1} behavior: {EXPLANATION_PREFIX} "
                f"{example.explanation}\n",
            )
            formatted_activation_records = format_activation_records(
                example.activation_records,
                calculate_max_activation(example.activation_records),
                start_indices=None,
            )
            prompt_builder.add_message(
                Role.ASSISTANT,
                f"Activations: {formatted_activation_records}\n\n",
            )

        prompt_builder.add_message(
            Role.SYSTEM,
            "Now, we're going predict the activation of a new neuron on a single token, "
            "following the same rules as the examples above. Activations still range from 0 to 10.",
        )
        single_token_example = self.few_shot_example_set.get_single_token_prediction_example()
        assert single_token_example.token_index_to_score is not None
        self._add_single_token_simulation_subprompt(
            prompt_builder,
            single_token_example.activation_records[0],
            len(few_shot_examples) + 1,
            explanation,
            token_index_to_score=single_token_example.token_index_to_score,
            end_of_prompt=False,
        )

        activation_record = ActivationRecord(
            tokens=list(tokens[: token_index_to_score + 1]),  # ActivationRecord expects List type.
            activations=[0.0] * len(tokens),
        )
        self._add_single_token_simulation_subprompt(
            prompt_builder,
            activation_record,
            len(few_shot_examples) + 2,
            explanation,
            token_index_to_score,
            end_of_prompt=True,
        )
        return prompt_builder.build(self.prompt_format, allow_extra_system_messages=True)


def _format_record_for_logprob_free_simulation(
    activation_record: ActivationRecord,
    include_activations: bool = False,
    max_activation: Optional[float] = None,
) -> str:
    response = ""
    if include_activations:
        assert max_activation is not None
        assert len(activation_record.tokens) == len(
            activation_record.activations
        ), f"{len(activation_record.tokens)=}, {len(activation_record.activations)=}"
        normalized_activations = normalize_activations(
            activation_record.activations, max_activation=max_activation
        )
    for i, token in enumerate(activation_record.tokens):
        # We use a weird unicode character here to make it easier to parse the response (can split on "༗\n").
        if include_activations:
            response += f"{token}\t{normalized_activations[i]}༗\n"
        else:
            response += f"{token}\t༗\n"
    return response


def _parse_no_logprobs_completion(
    completion: str,
    tokens: Sequence[str],
) -> Sequence[int]:
    """
    Parse a completion into a list of simulated activations. If the model did not faithfully
    reproduce the token sequence, return a list of 0s. If the model's activation for a token
    is not an integer betwee 0 and 10, substitute 0.

    Args:
        completion: completion from the API
        tokens: list of tokens as strings in the sequence where the neuron is being simulated
    """
    zero_prediction = [0] * len(tokens)
    token_lines = completion.strip("\n").split("༗\n")
    start_line_index = None
    for i, token_line in enumerate(token_lines):
        if token_line.startswith(f"{tokens[0]}\t"):
            start_line_index = i
            break

    # If we didn't find the first token, or if the number of lines in the completion doesn't match
    # the number of tokens, return a list of 0s.
    if start_line_index is None or len(token_lines) - start_line_index != len(tokens):
        return zero_prediction
    predicted_activations = []
    for i, token_line in enumerate(token_lines[start_line_index:]):
        if not token_line.startswith(f"{tokens[i]}\t"):
            return zero_prediction
        predicted_activation = token_line.split("\t")[1]
        if predicted_activation not in VALID_ACTIVATION_TOKENS:
            predicted_activations.append(0)
        else:
            predicted_activations.append(int(predicted_activation))
    return predicted_activations


class LogprobFreeExplanationTokenSimulator(NeuronSimulator):
    """
    Simulate neuron behavior based on an explanation.

    Unlike ExplanationNeuronSimulator and ExplanationTokenByTokenSimulator, this class does not rely on
    logprobs to calculate expected activations. Instead, it uses a few-shot prompt that displays all of the
    tokens at once, and request that the model repeat the tokens with the activations appended. Sampling
    is with temperature = 0. Thus, the activations are deterministic. Also, each activation for a token
    is a function of all the activations that came previously and all of the tokens in the sequence, not
    just the current and previous tokens. In the case where the model does not faithfully reproduce the
    token sequence, the simulator will return a response where every predicted activation is 0. Example prompt as follows:

    Explanation: Explanation 1

    Sequence 1 Tokens Without Activations:

    A\t_
    B\t_
    C\t_

    Sequence 1 Tokens With Activations:

    A\t4_
    B\t10_
    C\t0_

    Sequence 2 Tokens Without Activations:

    D\t_
    E\t_
    F\t_

    Sequence 2 Tokens With Activations:

    D\t3_
    E\t6_
    F\t9_

    Explanation: Explanation 2

    Sequence 1 Tokens Without Activations:

    G\t_
    H\t_
    I\t_

    Sequence 1 Tokens With Activations:
    <start sampling here>

    G\t2_
    H\t0_
    I\t3_

    """

    def __init__(
        self,
        model_name: str,
        explanation: str,
        max_concurrent: Optional[int] = 10,
        few_shot_example_set: FewShotExampleSet = FewShotExampleSet.NEWER,
        prompt_format: PromptFormat = PromptFormat.HARMONY_V4,
        cache: bool = False,
    ):
        assert (
            few_shot_example_set != FewShotExampleSet.ORIGINAL
        ), "This simulator doesn't support the ORIGINAL few-shot example set."
        self.api_client = ApiClient(
            model_name=model_name, max_concurrent=max_concurrent, cache=cache
        )
        self.explanation = explanation
        self.few_shot_example_set = few_shot_example_set
        self.prompt_format = prompt_format

    async def simulate(
        self,
        tokens: Sequence[str],
    ) -> SequenceSimulation:
        prompt = self._make_simulation_prompt(
            tokens,
            self.explanation,
        )
        response = await self.api_client.make_request(
            messages=prompt, max_tokens=1000
        )
        assert len(response["choices"]) == 1

        choice = response["choices"][0]
        if self.prompt_format == PromptFormat.HARMONY_V4:
            completion = choice["message"]["content"]
        elif self.prompt_format in [PromptFormat.NONE, PromptFormat.INSTRUCTION_FOLLOWING]:
            completion = choice["text"]
        else:
            raise ValueError(f"Unhandled prompt format {self.prompt_format}")

        predicted_activations = _parse_no_logprobs_completion(completion, tokens)

        result = SequenceSimulation(
            activation_scale=ActivationScale.SIMULATED_NORMALIZED_ACTIVATIONS,
            expected_activations=predicted_activations,
            # Since the predicted activation is just a sampled token, we don't have a distribution.
            distribution_values=[],
            distribution_probabilities=[],
            tokens=list(tokens),  # SequenceSimulation expects List type
        )
        logger.debug("result in score_explanation_by_activations is %s", result)
        return result

    def _make_simulation_prompt(
        self,
        tokens: Sequence[str],
        explanation: str,
    ) -> Union[str, list[HarmonyMessage]]:
        """Make a few-shot prompt for predicting the neuron's activations on a sequence."""
        assert explanation != ""
        prompt_builder = PromptBuilder(allow_extra_system_messages=True)
        prompt_builder.add_message(
            Role.SYSTEM,
            """We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at  an explanation of what the neuron does, and try to predict its activations on a particular token.

The activation format is token<tab>activation, and activations range from 0 to 10. Most activations will be 0.
For each sequence, you will see the tokens in the sequence where the activations are left blank. You will print the exact same tokens verbatim, but with the activations filled in according to the explanation.
""",
        )

        few_shot_examples = self.few_shot_example_set.get_examples()
        for i, example in enumerate(few_shot_examples):
            few_shot_example_max_activation = calculate_max_activation(example.activation_records)

            prompt_builder.add_message(
                Role.USER,
                f"Neuron {i + 1}\nExplanation of neuron {i + 1} behavior: {EXPLANATION_PREFIX} "
                f"{example.explanation}\n\n"
                f"Sequence 1 Tokens without Activations:\n{_format_record_for_logprob_free_simulation(example.activation_records[0], include_activations=False)}\n\n"
                f"Sequence 1 Tokens with Activations:\n",
            )
            prompt_builder.add_message(
                Role.ASSISTANT,
                f"{_format_record_for_logprob_free_simulation(example.activation_records[0], include_activations=True, max_activation=few_shot_example_max_activation)}\n\n",
            )

            for record_index, record in enumerate(example.activation_records[1:]):
                prompt_builder.add_message(
                    Role.USER,
                    f"Sequence {record_index + 2} Tokens without Activations:\n{_format_record_for_logprob_free_simulation(record, include_activations=False)}\n\n"
                    f"Sequence {record_index + 2} Tokens with Activations:\n",
                )
                prompt_builder.add_message(
                    Role.ASSISTANT,
                    f"{_format_record_for_logprob_free_simulation(record, include_activations=True, max_activation=few_shot_example_max_activation)}\n\n",
                )

        neuron_index = len(few_shot_examples) + 1
        prompt_builder.add_message(
            Role.USER,
            f"Neuron {neuron_index}\nExplanation of neuron {neuron_index} behavior: {EXPLANATION_PREFIX} "
            f"{explanation}\n\n"
            f"Sequence 1 Tokens without Activations:\n{_format_record_for_logprob_free_simulation(ActivationRecord(tokens=tokens, activations=[]), include_activations=False)}\n\n"
            f"Sequence 1 Tokens with Activations:\n",
        )
        return prompt_builder.build(self.prompt_format)
