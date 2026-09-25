from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from openrouter.errors import OpenRouterError

from langchain_core.tools import tool

from lgraph.api import create_app
from lgraph.config import Settings
from lgraph.prompts import DEFAULT_RAG_SYSTEM_PROMPT
from lgraph.rag.store import PaperStore

from .conftest import FakeEmbeddings, RaisingChatModel, fake_model, make_chunks, make_paper

REASONING = [{"type": "reasoning.text", "text": "t", "format": "unknown"}]


def _client(settings: Settings, model, tools=()) -> TestClient:
    app = create_app(settings, model=model, tools=list(tools))
    return TestClient(app, raise_server_exceptions=False)


def _source(n: int, chunk: str) -> dict:
    return {
        "n": n,
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "year": 2017,
        "doi": None,
        "arxiv_id": "1706.03762",
        "section": "3.2 Attention",
        "page": 4,
        "chunk_id": chunk,
    }


@tool(response_format="content_and_artifact")
def lookup(query: str) -> tuple[str, list[dict]]:
    """Look up papers."""
    return f"excerpts for {query}", [_source(1, "arxiv:1706.03762#0003"), _source(2, "arxiv:1706.03762#0004")]


def _tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def test_health_reports_model_without_calling_upstream(settings: Settings) -> None:
    model = fake_model()  # any call would raise StopIteration
    with _client(settings, model) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": "test/model"}
    assert model.calls == []


def test_chat_returns_history_plus_assistant_reply(settings: Settings) -> None:
    model = fake_model(AIMessage(content="hello", additional_kwargs={"reasoning_details": REASONING}))
    with _client(settings, model) as c:
        r = c.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert r.status_code == 200
    assert r.json() == {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning_details": REASONING},
        ],
        "sources": [],
    }


def test_chat_forwards_prior_reasoning_details_to_model(settings: Settings) -> None:
    model = fake_model("second")
    history = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "first", "reasoning_details": REASONING},
        {"role": "user", "content": "two"},
    ]
    with _client(settings, model) as c:
        r = c.post("/chat", json={"messages": history})

    assert r.status_code == 200
    [call] = model.calls
    assert call[1].additional_kwargs["reasoning_details"] == REASONING
    assert len(r.json()["messages"]) == 4


def test_chat_uses_system_prompt_from_settings() -> None:
    settings = Settings(api_key="k", system_prompt="Be terse.")
    model = fake_model("ok")
    with _client(settings, model) as c:
        c.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    [call] = model.calls
    assert call[0].content == "Be terse."


def test_chat_rejects_empty_message_list(settings: Settings) -> None:
    with _client(settings, fake_model()) as c:
        r = c.post("/chat", json={"messages": []})
    assert r.status_code == 422


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_upstream"),
    [
        (httpx.ReadTimeout("slow"), 504, None),
        (OpenRouterError("rate", raw_response=httpx.Response(429, text="slow down")), 429, 429),
        (OpenRouterError("boom", raw_response=httpx.Response(503, text="down")), 502, 503),
        (ValueError("OpenRouter API returned an error: nope"), 502, None),
    ],
)
def test_upstream_failures_are_translated(
    settings: Settings, error: Exception, expected_status: int, expected_upstream: int | None
) -> None:
    with _client(settings, RaisingChatModel(error=error)) as c:
        r = c.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert r.status_code == expected_status
    body = r.json()
    assert set(body) == {"detail", "upstream_status"}
    assert body["upstream_status"] == expected_upstream


def test_unmapped_errors_surface_as_500(settings: Settings) -> None:
    with _client(settings, RaisingChatModel(error=RuntimeError("bug"))) as c:
        r = c.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 500


def test_chat_runs_tools_and_reports_sources(settings: Settings) -> None:
    model = fake_model(
        AIMessage(content="", tool_calls=[_tool_call("lookup", {"query": "attention"}, "call-1")]),
        AIMessage(content="Multi-head attention... (Vaswani et al., 2017, p. 4)"),
    )
    with _client(settings, model, tools=[lookup]) as c:
        r = c.post("/chat", json={"messages": [{"role": "user", "content": "explain attention"}]})

    assert r.status_code == 200
    body = r.json()
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert body["messages"][1]["tool_calls"][0]["function"]["name"] == "lookup"
    assert body["messages"][2]["tool_call_id"] == "call-1"
    assert body["messages"][2]["content"] == "excerpts for attention"
    assert [s["chunk_id"] for s in body["sources"]] == ["arxiv:1706.03762#0003", "arxiv:1706.03762#0004"]
    assert [s["n"] for s in body["sources"]] == [1, 2]
    assert body["sources"][0]["page"] == 4


def test_sources_are_deduplicated_across_tool_calls(settings: Settings) -> None:
    model = fake_model(
        AIMessage(content="", tool_calls=[
            _tool_call("lookup", {"query": "a"}, "call-1"),
            _tool_call("lookup", {"query": "b"}, "call-2"),
        ]),
        AIMessage(content="done"),
    )
    with _client(settings, model, tools=[lookup]) as c:
        r = c.post("/chat", json={"messages": [{"role": "user", "content": "q"}]})

    assert r.status_code == 200
    chunk_ids = [s["chunk_id"] for s in r.json()["sources"]]
    assert chunk_ids == ["arxiv:1706.03762#0003", "arxiv:1706.03762#0004"]
    assert [s["n"] for s in r.json()["sources"]] == [1, 2]


def test_prior_tool_turns_round_trip_through_chat(settings: Settings) -> None:
    model = fake_model("follow-up answer")
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "type": "function", "id": "call-1",
            "function": {"name": "lookup", "arguments": '{"query": "a"}'},
        }]},
        {"role": "tool", "name": "lookup", "tool_call_id": "call-1", "content": "excerpts"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "and then?"},
    ]
    with _client(settings, model, tools=[lookup]) as c:
        r = c.post("/chat", json={"messages": history})

    assert r.status_code == 200
    body = r.json()
    assert body["messages"][:5] == history
    assert body["messages"][5]["content"] == "follow-up answer"
    # Tool results echoed from history are not this turn's retrievals.
    assert body["sources"] == []
    [call] = model.calls
    assert call[1].tool_calls[0]["id"] == "call-1"
    assert call[2].tool_call_id == "call-1"


# --- Retrieval wiring --------------------------------------------------------


def _rag_settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(api_key="k", model="test/model", data_dir=tmp_path, rag_enabled=True,
                    embedding_dimensions=8, **overrides)


def _seed(tmp_path: Path, emb: FakeEmbeddings) -> None:
    store = PaperStore.open(tmp_path / "lancedb", dimensions=emb.dimensions)
    paper = make_paper()
    chunks = make_chunks(paper, "multi-head attention mechanism", "positional encoding", section="3.2 Attention")
    store.upsert(paper, chunks, emb.embed_documents([c.text for c in chunks]))


def test_rag_enabled_wires_search_tool_and_default_prompt(tmp_path: Path) -> None:
    emb = FakeEmbeddings(8)
    _seed(tmp_path, emb)
    model = fake_model(
        AIMessage(content="", tool_calls=[_tool_call("search_publications", {"query": "multi-head attention"}, "call-1")]),
        AIMessage(content="Answer (Vaswani et al., 2017, p. 1)."),
    )
    app = create_app(_rag_settings(tmp_path), model=model, embeddings=emb)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/chat", json={"messages": [{"role": "user", "content": "what is attention?"}]})

    assert r.status_code == 200
    body = r.json()
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert "[1] Vaswani et al. (2017)" in body["messages"][2]["content"]
    assert body["sources"][0]["chunk_id"] == "arxiv:1706.03762#0000"
    [call] = model.calls[:1]
    assert call[0].content == DEFAULT_RAG_SYSTEM_PROMPT


def test_rag_enabled_respects_explicit_system_prompt(tmp_path: Path) -> None:
    emb = FakeEmbeddings(8)
    model = fake_model("ok")
    app = create_app(_rag_settings(tmp_path, system_prompt="Custom."), model=model, embeddings=emb)
    with TestClient(app) as c:
        c.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert model.calls[0][0].content == "Custom."


def test_papers_endpoint_lists_corpus(tmp_path: Path) -> None:
    emb = FakeEmbeddings(8)
    _seed(tmp_path, emb)
    app = create_app(_rag_settings(tmp_path), model=fake_model(), embeddings=emb)
    with TestClient(app) as c:
        r = c.get("/papers")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["papers"][0]["title"] == "Attention Is All You Need"
    assert body["papers"][0]["chunk_count"] == 2


def test_papers_endpoint_when_rag_disabled(settings: Settings) -> None:
    with _client(settings, fake_model()) as c:
        r = c.get("/papers")
    assert r.status_code == 200
    assert r.json() == {"enabled": False, "papers": []}


def test_rag_disabled_has_no_tools_and_no_default_prompt(settings: Settings) -> None:
    model = fake_model("ok")
    with _client(settings, model) as c:
        r = c.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.json()["sources"] == []
    assert [type(m).__name__ for m in model.calls[0]] == ["HumanMessage"]


@pytest.mark.parametrize(
    "bad",
    [
        [{"role": "tool", "content": "x"}],  # tool message without tool_call_id
        [{"role": "bogus", "content": "x"}],
        [{"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]}],
        [{"role": "assistant", "content": "", "tool_calls": [{
            "type": "function", "id": "c", "function": {"name": "f", "arguments": "not json"}}]}],
    ],
)
def test_malformed_messages_are_422_not_500(settings: Settings, bad: list) -> None:
    with _client(settings, fake_model("unused")) as c:
        r = c.post("/chat", json={"messages": bad})
    assert r.status_code == 422
    assert "Invalid message list" in r.json()["detail"]


# --- /results static files ---------------------------------------------------


def test_results_endpoint_serves_html_file(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "report.html").write_text("<h1>Report</h1>")
    settings = Settings(api_key="k", model="test/model", data_dir=tmp_path)
    with _client(settings, fake_model()) as c:
        r = c.get("/results/report.html")
    assert r.status_code == 200
    assert "<h1>Report</h1>" in r.text


def test_results_endpoint_404_for_missing_file(tmp_path: Path) -> None:
    settings = Settings(api_key="k", model="test/model", data_dir=tmp_path)
    with _client(settings, fake_model()) as c:
        r = c.get("/results/missing.html")
    assert r.status_code == 404


def test_results_endpoint_creates_directory_when_absent(tmp_path: Path) -> None:
    settings = Settings(api_key="k", model="test/model", data_dir=tmp_path)
    with _client(settings, fake_model()):
        pass
    assert (tmp_path / "results").is_dir()
