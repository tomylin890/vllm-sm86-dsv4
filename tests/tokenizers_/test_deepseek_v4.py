# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.entrypoints.chat_utils import parse_chat_messages
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer
from vllm.tokenizers.registry import TokenizerRegistry

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "deepseek_v4"


class FakeHfTokenizer:
    vocab_size = 100

    def get_added_vocab(self) -> dict[str, int]:
        return {"</think>": 100}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> list[int]:
        self.last_encode = (text, add_special_tokens, kwargs)
        return [len(text)]


def _tokenizer():
    return get_deepseek_v4_tokenizer(FakeHfTokenizer())


def _model_config():
    return SimpleNamespace(
        multimodal_config=None,
        allowed_local_media_path="",
        allowed_media_domains=None,
        enable_prompt_embeds=False,
    )


def _load_reference_case(case_id: int):
    data = json.loads((FIXTURES_DIR / f"test_input_{case_id}.json").read_text())
    if isinstance(data, dict):
        return data["messages"], data.get("tools")
    return data, None


def _render_reference_case(case_id: int, **kwargs):
    messages, tools = _load_reference_case(case_id)
    conversation, _, _ = parse_chat_messages(
        messages,
        _model_config(),
        content_format="string",
    )
    return _tokenizer().apply_chat_template(
        conversation=conversation,
        messages=messages,
        tools=tools,
        tokenize=False,
        **kwargs,
    )


def test_deepseek_v4_tokenizer_registered():
    assert TokenizerRegistry.load_tokenizer_cls("deepseek_v4").__name__ == (
        "DeepseekV4Tokenizer"
    )
    assert RENDERER_REGISTRY.load_renderer_cls("deepseek_v4").__name__ == (
        "DeepseekV4Renderer"
    )


def test_deepseek_v4_thinking_is_opt_in_when_no_kwargs():
    """No thinking key -> chat mode. DELIBERATE DIVERGENCE from upstream.

    Upstream #50580 made a request carrying neither `thinking` nor
    `enable_thinking` default to thinking mode here. The reasoning parser
    derives the same decision independently (vllm/parser/deepseek_v4.py reads
    `chat_template_kwargs`) and was not changed with it, so the two disagree
    for exactly those requests: this renderer primes the prompt with
    `<think>`, the parser starts in CONTENT, and the model's closing
    `</think>` hits the (CONTENT, THINK_END) transition that absorbs it
    without emitting REASONING_END -- the reasoning is generated, never
    routed, and is served as the answer.

    Verified on the 8x3090 deployment: a request with no chat_template_kwargs
    came back with 78 characters of deliberation in `content` and an empty
    `reasoning`, byte-for-byte the text that the same request returns under
    `reasoning` when thinking=true is passed.

    Keeping thinking opt-in makes both sides agree on every input. A
    deployment that wants thinking by default states it explicitly
    (--default-chat-template-kwargs '{"thinking":true}'), which the parser
    can see too. If a future upstream merge re-flips this default, this test
    fails -- and the parser must be flipped in the same commit.
    """
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
    )

    assert "Reasoning Effort:" not in prompt
    assert prompt.endswith("<｜Assistant｜></think>")


def test_deepseek_v4_explicit_thinking_flags_still_work():
    """The shapes our clients actually send: both are honoured, and both
    agree with the parser's own reading of the same kwargs."""
    on = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}], tokenize=False, thinking=True
    )
    assert on.endswith("<｜Assistant｜><think>")
    assert on.startswith("<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum")

    off = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}], tokenize=False, thinking=False
    )
    assert off.endswith("<｜Assistant｜></think>")
    assert "Reasoning Effort:" not in off


@pytest.mark.parametrize("kwargs", [{"thinking": True}, {"enable_thinking": True}])
def test_deepseek_v4_enables_thinking_with_compatible_kwargs(kwargs):
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        **kwargs,
    )

    assert prompt.startswith(
        "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"
    )
    assert prompt.endswith("<｜Assistant｜><think>")


@pytest.mark.parametrize("kwargs", [{"thinking": False}, {"enable_thinking": False}])
def test_deepseek_v4_explicitly_disables_thinking(kwargs):
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        **kwargs,
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>")


def test_deepseek_v4_uses_v4_tool_prompt_from_request_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Weather?"}],
        tools=tools,
        tokenize=False,
        # Explicit: thinking is opt-in on this fork (see
        # test_deepseek_v4_thinking_is_opt_in_when_no_kwargs), so the effort
        # prefix and the `<think>` ending this test also checks require the
        # flag. The subject of the test -- the V4 tool prompt -- is unchanged.
        thinking=True,
    )

    assert "## Tools" in prompt
    assert "<｜DSML｜tool_calls>" in prompt
    assert "</｜DSML｜tool_calls>" in prompt
    assert "function_calls" not in prompt
    assert '"name": "get_weather"' in prompt
    assert prompt.startswith(
        "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"
    )
    assert prompt.endswith("<｜User｜>Weather?<｜Assistant｜><think>")


def test_deepseek_v4_renders_parsed_history_tool_arguments():
    messages = [
        {"role": "user", "content": "List the repo"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "str_replace_editor",
                        "arguments": '{"command": "view", "path": "/testbed"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "file list",
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "description": "Edit files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "required": ["command", "path"],
                },
            },
        }
    ]
    conversation, _, _ = parse_chat_messages(
        messages,
        _model_config(),
        content_format="string",
    )

    prompt = _tokenizer().apply_chat_template(
        conversation=conversation,
        messages=messages,
        tools=tools,
        tokenize=False,
    )

    assert '<｜DSML｜parameter name="command" string="true">view' in prompt
    assert '<｜DSML｜parameter name="path" string="true">/testbed' in prompt
    assert 'parameter name="arguments"' not in prompt


@pytest.mark.parametrize(
    ("reasoning_effort", "expected_prefix"),
    [
        ("low", "<｜begin▁of▁sentence｜><｜User｜>Hello"),
        ("high", "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"),
        ("max", "<｜begin▁of▁sentence｜>Reasoning Effort: Beyond maximum"),
    ],
)
def test_deepseek_v4_renders_0731_reasoning_effort_prompts(
    reasoning_effort, expected_prefix
):
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort=reasoning_effort,
    )

    assert prompt.endswith("<｜Assistant｜><think>")
    assert prompt.startswith(expected_prefix)


def test_deepseek_v4_none_reasoning_effort_disables_thinking():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="none",
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>")


@pytest.mark.parametrize(
    ("reasoning_effort", "expected_mode", "expected_effort"),
    [
        ("none", "chat", None),
        ("minimal", "thinking", "low"),
        ("low", "thinking", "low"),
        ("medium", "thinking", "low"),
        ("high", "thinking", "high"),
        ("xhigh", "thinking", "high"),
        ("max", "thinking", "max"),
        ("unexpected", "thinking", "high"),
    ],
)
def test_deepseek_v4_maps_compatible_thinking_reasoning_effort_values(
    monkeypatch: pytest.MonkeyPatch,
    reasoning_effort,
    expected_mode,
    expected_effort,
):
    captured_kwargs = []

    def fake_encode_messages(messages, **kwargs):
        captured_kwargs.append(kwargs)
        return "prompt"

    monkeypatch.setattr(
        "vllm.tokenizers.deepseek_v4.encode_messages",
        fake_encode_messages,
    )

    _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort=reasoning_effort,
    )

    assert captured_kwargs[-1]["thinking_mode"] == expected_mode
    assert captured_kwargs[-1]["reasoning_effort"] == expected_effort


def test_deepseek_v4_renders_0731_max_reasoning_effort():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="max",
    )

    assert prompt.startswith("<｜begin▁of▁sentence｜>Reasoning Effort: Beyond maximum")


def test_deepseek_v4_maps_xhigh_to_high_reasoning_effort():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="xhigh",
    )

    assert prompt.startswith(
        "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"
    )


@pytest.mark.parametrize(
    ("case_id", "kwargs"),
    [
        (1, {"thinking": True, "reasoning_effort": "low"}),
        (2, {"thinking": True, "reasoning_effort": "low"}),
        (3, {"thinking": True, "reasoning_effort": "low"}),
        (4, {"thinking": False}),
    ],
)
def test_deepseek_v4_matches_reference_golden_fixtures(case_id, kwargs):
    prompt = _render_reference_case(case_id, **kwargs)

    expected = (FIXTURES_DIR / f"test_output_{case_id}.txt").read_text()
    assert prompt == expected


def test_history_without_reasoning_renders_no_empty_think_block():
    """A history assistant turn whose reasoning was not echoed back must render
    the no-thinking form (a bare closer), never an empty ``<think></think>``.

    The opener comes from the preceding message's transition and the closer
    from the assistant message, so an unguarded pair produces a turn that
    "opened thinking and wrote nothing" -- an in-context demonstration
    DeepSeek-V4 imitates, emitting ``</think>`` as its first token on the next
    turn (answer lands in content, reasoning comes back empty). Stock
    OpenAI-compatible clients never echo ``reasoning``, so this is the default
    shape of every multi-turn tool conversation.
    """
    from vllm.tokenizers.deepseek_v4_encoding import encode_messages

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    tool_call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
    }

    def convo(reasoning: str | None):
        assistant = {"role": "assistant", "content": "", "tool_calls": [tool_call]}
        if reasoning is not None:
            assistant["reasoning"] = reasoning
        return [
            {"role": "system", "content": "agent", "tools": tools},
            {"role": "user", "content": "read it"},
            assistant,
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "user", "content": "summarize"},
        ]

    without = encode_messages(convo(None), thinking_mode="thinking")
    assert "<think></think>" not in without
    assert "</think></think>" not in without
    # The no-thinking form: assistant turn opens with a bare closer.
    assert "<｜Assistant｜></think>" in without
    # The live generation position still primes thinking.
    assert without.endswith("<think>")

    # Supplied reasoning is preserved verbatim inside a real block.
    with_reasoning = encode_messages(convo("plan it"), thinking_mode="thinking")
    assert "<think>plan it</think>" in with_reasoning
    assert "<think></think>" not in with_reasoning
    assert with_reasoning.endswith("<think>")
