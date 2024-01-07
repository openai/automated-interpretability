from __future__ import annotations

from enum import Enum
from typing import TypedDict, Union

import tiktoken

HarmonyMessage = TypedDict(
    "HarmonyMessage",
    {
        "role": str,
        "content": str,
    },
)


class PromptFormat(str, Enum):
    """
    Different ways of formatting the components of a prompt into the format accepted by the relevant
    API server endpoint.
    """

    NONE = "none"
    """Suitable for use with models that don't use special tokens for instructions."""
    INSTRUCTION_FOLLOWING = "instruction_following"
    """Suitable for IF models that use <|endofprompt|>."""
    HARMONY_V4 = "harmony_v4"
    """
    Suitable for Harmony models that use a structured turn-taking role+content format. Generates a
    list of HarmonyMessage dicts that can be sent to the /chat/completions endpoint.
    """

    @classmethod
    def from_string(cls, s: str) -> PromptFormat:
        for prompt_format in cls:
            if prompt_format.value == s:
                return prompt_format
        raise ValueError(f"{s} is not a valid PromptFormat")


class Role(str, Enum):
    """See https://platform.openai.com/docs/guides/chat"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class PromptBuilder:
    """Class for accumulating components of a prompt and then formatting them into an output."""

    def __init__(self) -> None:
        self._messages: list[HarmonyMessage] = []

    def add_message(self, role: Role, message: str) -> None:
        self._messages.append(HarmonyMessage(role=role, content=message))

    def prompt_length_in_tokens(self, prompt_format: PromptFormat) -> int:
        # TODO(sbills): Make the model/encoding configurable. This implementation assumes GPT-4.
        encoding = tiktoken.get_encoding("cl100k_base")
        if prompt_format == PromptFormat.HARMONY_V4:
            # Approximately-correct implementation adapted from this documentation:
            # https://platform.openai.com/docs/guides/chat/introduction
            num_tokens = 0
            for message in self._messages:
                num_tokens += (
                    4  # every message follows <|im_start|>{role/name}\n{content}<|im_end|>\n
                )
                num_tokens += len(encoding.encode(message["content"], allowed_special="all"))
            num_tokens += 2  # every reply is primed with <|im_start|>assistant
            return num_tokens
        else:
            prompt_str = self.build(prompt_format)
            assert isinstance(prompt_str, str)
            return len(encoding.encode(prompt_str, allowed_special="all"))

    def build(
        self, prompt_format: PromptFormat, *, allow_extra_system_messages: bool = False
    ) -> Union[str, list[HarmonyMessage]]:
        """
        Validates the messages added so far (reasonable alternation of assistant vs. user, etc.)
        and returns either a regular string (maybe with <|endofprompt|> tokens) or a list of
        HarmonyMessages suitable for use with the /chat/completions endpoint.

        The `allow_extra_system_messages` parameter allows the caller to specify that the prompt
        should be allowed to contain system messages after the very first one.
        """
        # Create a deep copy of the messages so we can modify it and so that the caller can't
        # modify the internal state of this object.
        messages = [message.copy() for message in self._messages]

        expected_next_role = Role.SYSTEM
        for message in messages:
            role = message["role"]
            assert role == expected_next_role or (
                allow_extra_system_messages and role == Role.SYSTEM
            ), f"Expected message from {expected_next_role} but got message from {role}"
            if role == Role.SYSTEM:
                expected_next_role = Role.USER
            elif role == Role.USER:
                expected_next_role = Role.ASSISTANT
            elif role == Role.ASSISTANT:
                expected_next_role = Role.USER

        if prompt_format == PromptFormat.INSTRUCTION_FOLLOWING:
            last_user_message = None
            for message in messages:
                if message["role"] == Role.USER:
                    last_user_message = message
            assert last_user_message is not None
            last_user_message["content"] += "<|endofprompt|>"

        if prompt_format == PromptFormat.HARMONY_V4:
            return messages
        elif prompt_format in [PromptFormat.NONE, PromptFormat.INSTRUCTION_FOLLOWING]:
            return "".join(message["content"] for message in messages)
        else:
            raise ValueError(f"Unknown prompt format: {prompt_format}")
