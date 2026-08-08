# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from typing import Any

from transformers import TokenizersBackend

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

from .deepseek_v4_encoding import encode_messages
from .hf import HfTokenizer, get_cached_tokenizer
from .protocol import TokenizerLike


def get_deepseek_v4_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """
    Wraps a tokenizer to use the custom DeepSeek V4 chat template encoding.
    """
    dsv4_tokenizer = copy.copy(tokenizer)

    added_vocab = tokenizer.get_added_vocab()
    added_vocab_size = len(added_vocab)
    tokenizer_vocab_size = tokenizer.vocab_size

    class _DeepseekV4Tokenizer(tokenizer.__class__):  # type: ignore
        def apply_chat_template(
            self,
            messages: list["ChatCompletionMessageParam"],
            tools: list[dict[str, Any]] | None = None,
            **kwargs,
        ) -> str | list[int]:
            thinking = kwargs.get("thinking")
            enable_thinking = kwargs.get("enable_thinking")
            # Thinking stays OPT-IN. Upstream #50580 flipped the no-key
            # default to enabled here, but the reasoning parser derives the
            # same decision independently -- from
            # `chat_template_kwargs.get("thinking")` in
            # vllm/parser/deepseek_v4.py -- and was not flipped with it. The
            # two then disagree for exactly the requests that carry neither
            # key: this side renders thinking mode and primes the prompt with
            # `<think>`, while the parser starts in CONTENT, where the model's
            # closing `</think>` hits the (CONTENT, THINK_END) transition that
            # absorbs it without emitting REASONING_END. The reasoning is
            # produced, is never routed, and lands in `content` -- verified on
            # the 8x3090 deployment: a request with no chat_template_kwargs
            # returned 78 characters of deliberation as its answer while
            # `reasoning` came back empty, byte-for-byte the text the same
            # request returns under `reasoning` when thinking=true is passed.
            #
            # Keeping opt-in makes the two sides agree again for every input.
            # Deployments that want thinking by default should say so
            # explicitly (--default-chat-template-kwargs '{"thinking":true}'),
            # which both this renderer and the parser can see. The rest of
            # #50580 -- the low/high/max effort table and the "high" default
            # for requests that DO enable thinking -- is kept.
            thinking_enabled = bool(thinking) or bool(enable_thinking)
            thinking_mode = "thinking" if thinking_enabled else "chat"

            conversation = kwargs.get("conversation", messages)
            messages = conversation.copy()
            if tools is not None and len(tools) > 0:
                messages.insert(0, {"role": "system"})
                messages[0]["tools"] = tools  # type: ignore[typeddict-unknown-key]

            reasoning_effort = kwargs.get("reasoning_effort")
            if not isinstance(reasoning_effort, str):
                reasoning_effort = "high" if thinking_enabled else None
            elif reasoning_effort == "none":
                thinking_mode = "chat"
                reasoning_effort = None
            elif reasoning_effort == "max":
                reasoning_effort = "max"
            elif reasoning_effort in ("low", "minimal", "medium"):
                reasoning_effort = "low"
            else:
                reasoning_effort = "high"

            encode_config = dict(
                thinking_mode=thinking_mode,
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=reasoning_effort,
            )

            prompt_str = encode_messages(messages, **encode_config)  # type: ignore

            if kwargs.get("tokenize", True):
                tokenizer_kwargs = {
                    k: kwargs[k] for k in ("truncation", "max_length") if k in kwargs
                }
                return self.encode(
                    prompt_str,
                    add_special_tokens=False,
                    **tokenizer_kwargs,
                )

            return prompt_str

        def num_special_tokens_to_add(self) -> int:
            return len(self.encode(""))

        def __len__(self) -> int:
            return tokenizer_vocab_size + added_vocab_size

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self):
            return get_deepseek_v4_tokenizer, (tokenizer,)

    _DeepseekV4Tokenizer.__name__ = f"DSV4{tokenizer.__class__.__name__}"

    dsv4_tokenizer.__class__ = _DeepseekV4Tokenizer
    return dsv4_tokenizer


class DeepseekV4Tokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> HfTokenizer:
        tokenizer = TokenizersBackend.from_pretrained(*args, **kwargs)
        return get_cached_tokenizer(get_deepseek_v4_tokenizer(tokenizer))
