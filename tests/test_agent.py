from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from lgraph.agent import build_agent

from .conftest import fake_model


async def test_agent_appends_one_assistant_message() -> None:
    model = fake_model(AIMessage(content="hello"))
    agent = build_agent(model)

    out = await agent.ainvoke({"messages": [HumanMessage(content="hi")]})

    assert len(out["messages"]) == 2
    assert isinstance(out["messages"][-1], AIMessage)
    assert out["messages"][-1].content == "hello"


async def test_agent_preserves_additional_kwargs_from_model() -> None:
    reasoning = [{"type": "reasoning.text", "text": "t"}]
    model = fake_model(AIMessage(content="x", additional_kwargs={"reasoning_details": reasoning}))
    agent = build_agent(model)

    out = await agent.ainvoke({"messages": [HumanMessage(content="hi")]})

    assert out["messages"][-1].additional_kwargs["reasoning_details"] == reasoning


async def test_system_prompt_is_prepended_for_the_model() -> None:
    model = fake_model("ok")
    agent = build_agent(model, system_prompt="Be terse.")

    out = await agent.ainvoke({"messages": [HumanMessage(content="hi")]})

    [call] = model.calls
    assert isinstance(call[0], SystemMessage)
    assert call[0].content == "Be terse."
    # The system prompt is a model-call concern; it must not leak into state.
    assert not any(isinstance(m, SystemMessage) for m in out["messages"])


async def test_no_system_prompt_means_no_system_message() -> None:
    model = fake_model("ok")
    agent = build_agent(model)

    await agent.ainvoke({"messages": [HumanMessage(content="hi")]})

    [call] = model.calls
    assert [type(m) for m in call] == [HumanMessage]
