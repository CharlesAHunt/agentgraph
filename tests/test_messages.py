from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from lgraph.messages import to_langchain, to_wire

REASONING = [{"type": "reasoning.text", "text": "thinking...", "format": "unknown"}]


def test_roles_convert_to_langchain_types() -> None:
    msgs = to_langchain(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    assert [type(m) for m in msgs] == [SystemMessage, HumanMessage, AIMessage]
    assert [m.content for m in msgs] == ["sys", "hi", "hello"]


def test_reasoning_details_land_in_additional_kwargs() -> None:
    [msg] = to_langchain(
        [{"role": "assistant", "content": "x", "reasoning_details": REASONING}]
    )
    assert isinstance(msg, AIMessage)
    assert msg.additional_kwargs["reasoning_details"] == REASONING


def test_none_valued_keys_are_dropped_on_input() -> None:
    [msg] = to_langchain([{"role": "assistant", "content": "x", "reasoning_details": None}])
    assert "reasoning_details" not in msg.additional_kwargs


def test_none_content_becomes_empty_string() -> None:
    [msg] = to_langchain([{"role": "user", "content": None}])
    assert msg.content == ""


def test_to_wire_roles_and_content() -> None:
    wire = to_wire(
        [
            SystemMessage(content="sys"),
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
            ToolMessage(content="42", tool_call_id="call-1"),
        ]
    )
    assert wire == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "tool", "content": "42", "tool_call_id": "call-1"},
    ]


def test_to_wire_flattens_content_blocks_to_text() -> None:
    [wire] = to_wire([AIMessage(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])])
    # Text blocks are joined the way the OpenAI converter does it: one per line.
    assert wire["content"] == "a\nb"


def test_reasoning_details_round_trip() -> None:
    original = [{"role": "assistant", "content": "x", "reasoning_details": REASONING}]
    assert to_wire(to_langchain(original)) == original


def test_to_wire_omits_reasoning_details_when_absent() -> None:
    [wire] = to_wire([AIMessage(content="x", additional_kwargs={"reasoning_content": "hidden"})])
    assert "reasoning_details" not in wire
    assert "reasoning_content" not in wire


TOOL_CALL = {"name": "search", "args": {"q": "x"}, "id": "call-1", "type": "tool_call"}
WIRE_TOOL_CALL = {
    "type": "function",
    "id": "call-1",
    "function": {"name": "search", "arguments": '{"q": "x"}'},
}


def test_to_wire_emits_tool_calls_and_reasoning_together() -> None:
    [wire] = to_wire(
        [AIMessage(content="", tool_calls=[TOOL_CALL], additional_kwargs={"reasoning_details": REASONING})]
    )
    assert wire == {
        "role": "assistant",
        "content": "",
        "tool_calls": [WIRE_TOOL_CALL],
        "reasoning_details": REASONING,
    }


def test_to_wire_tool_message_carries_name_and_call_id() -> None:
    [wire] = to_wire([ToolMessage(content="42", tool_call_id="call-1", name="search")])
    assert wire == {"role": "tool", "name": "search", "tool_call_id": "call-1", "content": "42"}


def test_tool_round_trip() -> None:
    original = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [WIRE_TOOL_CALL], "reasoning_details": REASONING},
        {"role": "tool", "name": "search", "tool_call_id": "call-1", "content": "42"},
        {"role": "assistant", "content": "answer"},
    ]
    msgs = to_langchain(original)
    assert isinstance(msgs[1], AIMessage)
    assert msgs[1].tool_calls == [TOOL_CALL]
    assert isinstance(msgs[2], ToolMessage)
    assert msgs[2].tool_call_id == "call-1"
    assert to_wire(msgs) == original
