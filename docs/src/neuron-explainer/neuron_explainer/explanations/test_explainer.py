import asyncio
from typing import Any

from neuron_explainer.explanations.explainer import (
    TokenActivationPairExplainer,
    TokenSpaceRepresentationExplainer,
)
from neuron_explainer.explanations.few_shot_examples import TEST_EXAMPLES, FewShotExampleSet
from neuron_explainer.explanations.prompt_builder import HarmonyMessage, PromptFormat, Role
from neuron_explainer.explanations.token_space_few_shot_examples import (
    TokenSpaceFewShotExampleSet,
)


def setup_module(unused_module: Any) -> None:
    # Make sure we have an event loop, since the attempt to create the Semaphore in
    # ResearchApiClient will fail without it.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


def test_if_formatting() -> None:
    expected_prompt = """We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is looking for. Don't list examples of words.

The activation format is token<tab>activation. Activation values range from 0 to 10. A neuron finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match.

Neuron 1
Activations:
<start>
a	10
b	0
c	0
<end>
<start>
d	0
e	10
f	0
<end>

Explanation of neuron 1 behavior: the main thing this neuron does is find vowels.

Neuron 2
Activations:
<start>
a	10
b	0
c	0
<end>
<start>
d	0
e	10
f	0
<end>

Explanation of neuron 2 behavior:<|endofprompt|> the main thing this neuron does is find"""

    explainer = TokenActivationPairExplainer(
        model_name="text-davinci-003",
        prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
        few_shot_example_set=FewShotExampleSet.TEST,
    )
    prompt = explainer.make_explanation_prompt(
        all_activation_records=TEST_EXAMPLES[0].activation_records,
        max_activation=1.0,
        max_tokens_for_completion=20,
    )

    assert prompt == expected_prompt


def test_harmony_format() -> None:
    expected_prompt = [
        HarmonyMessage(
            role=Role.SYSTEM,
            content="""We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is looking for. Don't list examples of words.

The activation format is token<tab>activation. Activation values range from 0 to 10. A neuron finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match.""",
        ),
        HarmonyMessage(
            role=Role.USER,
            content="""

Neuron 1
Activations:
<start>
a	10
b	0
c	0
<end>
<start>
d	0
e	10
f	0
<end>

Explanation of neuron 1 behavior: the main thing this neuron does is find""",
        ),
        HarmonyMessage(
            role=Role.ASSISTANT,
            content=" vowels.",
        ),
        HarmonyMessage(
            role=Role.USER,
            content="""

Neuron 2
Activations:
<start>
a	10
b	0
c	0
<end>
<start>
d	0
e	10
f	0
<end>

Explanation of neuron 2 behavior: the main thing this neuron does is find""",
        ),
    ]

    explainer = TokenActivationPairExplainer(
        model_name="gpt-4",
        prompt_format=PromptFormat.HARMONY_V4,
        few_shot_example_set=FewShotExampleSet.TEST,
    )
    prompt = explainer.make_explanation_prompt(
        all_activation_records=TEST_EXAMPLES[0].activation_records,
        max_activation=1.0,
        max_tokens_for_completion=20,
    )

    assert isinstance(prompt, list)
    assert isinstance(prompt[0], dict)  # Really a HarmonyMessage
    for actual_message, expected_message in zip(prompt, expected_prompt):
        assert actual_message["role"] == expected_message["role"]
        assert actual_message["content"] == expected_message["content"]
    assert prompt == expected_prompt


def test_token_space_explainer_if_formatting() -> None:
    expected_prompt = """We're studying neurons in a neural network. Each neuron looks for some particular kind of token (which can be a word, or part of a word). Look at the tokens the neuron activates for (listed below) and summarize in a single sentence what the neuron is looking for. Don't list examples of words.



Tokens:
'these', ' are', ' tokens'

Explanation:
This neuron is looking for this is a test explanation.



Tokens:
'foo', 'bar', 'baz'

Explanation:
<|endofprompt|>This neuron is looking for"""

    explainer = TokenSpaceRepresentationExplainer(
        model_name="text-davinci-002",
        prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
        use_few_shot=True,
        few_shot_example_set=TokenSpaceFewShotExampleSet.TEST,
    )
    prompt = explainer.make_explanation_prompt(
        tokens=["foo", "bar", "baz"],
        max_tokens_for_completion=20,
    )

    assert prompt == expected_prompt


def test_token_space_explainer_harmony_formatting() -> None:
    expected_prompt = [
        HarmonyMessage(
            role=Role.SYSTEM,
            content="We're studying neurons in a neural network. Each neuron looks for some particular kind of token (which can be a word, or part of a word). Look at the tokens the neuron activates for (listed below) and summarize in a single sentence what the neuron is looking for. Don't list examples of words.",
        ),
        HarmonyMessage(
            role=Role.USER,
            content="""



Tokens:
'these', ' are', ' tokens'

Explanation:
This neuron is looking for""",
        ),
        HarmonyMessage(
            role=Role.ASSISTANT,
            content=" this is a test explanation.",
        ),
        HarmonyMessage(
            role=Role.USER,
            content="""



Tokens:
'foo', 'bar', 'baz'

Explanation:
This neuron is looking for""",
        ),
    ]

    explainer = TokenSpaceRepresentationExplainer(
        model_name="gpt-4",
        prompt_format=PromptFormat.HARMONY_V4,
        use_few_shot=True,
        few_shot_example_set=TokenSpaceFewShotExampleSet.TEST,
    )
    prompt = explainer.make_explanation_prompt(
        tokens=["foo", "bar", "baz"],
        max_tokens_for_completion=20,
    )

    assert isinstance(prompt, list)
    assert isinstance(prompt[0], dict)  # Really a HarmonyMessage
    for actual_message, expected_message in zip(prompt, expected_prompt):
        assert actual_message["role"] == expected_message["role"]
        assert actual_message["content"] == expected_message["content"]
    assert prompt == expected_prompt
