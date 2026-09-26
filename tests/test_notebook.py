from __future__ import annotations

import ast
import asyncio
import base64
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from lgraph.notebook import sandbox as sb
from lgraph.notebook.sandbox import (
    EXECUTOR,
    KernelPool,
    Sandbox,
    SandboxError,
    container_command,
    detect_runtime,
    make_cell,
)
from lgraph.notebook.tool import summarize

PNG = base64.b64encode(sb.PNG_SIGNATURE + b"\x00" * 16).decode()


# --- commands ---------------------------------------------------------------------


def test_container_command_isolates_the_sandbox() -> None:
    argv = container_command(["distrobox-host-exec", "podman"], "lgraph-sandbox", "lgraph-py-abc")
    assert argv[:3] == ["distrobox-host-exec", "podman", "run"]
    joined = " ".join(argv)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges",
                 "--memory 1g", "--memory-swap 1g", "--pids-limit 256", "--pull never", "--rm -i"):
        assert flag in joined
    assert "-v" not in argv and "--volume" not in argv and "--mount" not in argv
    assert argv[-1] == "lgraph-sandbox"


def test_runtime_uses_host_podman_inside_a_distrobox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTAINER_ID", "fedora-box")
    monkeypatch.setattr(sb.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    assert detect_runtime() == ["distrobox-host-exec", "podman"]


def test_runtime_falls_back_to_podman_then_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTAINER_ID", raising=False)
    monkeypatch.setattr(sb.Path, "exists", lambda self: False)
    monkeypatch.setattr(sb.shutil, "which", lambda cmd: "/usr/bin/docker" if cmd == "docker" else None)
    assert detect_runtime() == ["docker"]


# --- output limits ------------------------------------------------------------------


def test_cell_keeps_valid_outputs_and_drops_the_rest() -> None:
    reply = {"status": "ok", "execution_count": 3, "outputs": [
        {"type": "stream", "name": "stdout", "text": "hi\n"},
        {"type": "latex", "latex": "$x^2$", "text": "x**2"},
        {"type": "image", "png": PNG, "text": "<Figure>"},
        {"type": "image", "png": base64.b64encode(b"GIF89a....").decode()},  # not a PNG
        {"type": "image", "png": "not base64 !!"},
        {"type": "html", "html": "<script>alert(1)</script>"},  # unknown type
        {"type": "error", "ename": "ValueError", "evalue": "bad", "traceback": "tb"},
    ]}
    cell = make_cell("x**2", reply, 12)
    assert [o["type"] for o in cell["outputs"]] == ["stream", "latex", "image", "error"]
    assert cell["outputs"][2]["png"] == PNG
    assert cell["status"] == "ok" and cell["execution_count"] == 3 and cell["code"] == "x**2"


def test_cell_bounds_text_images_and_status() -> None:
    reply = {"status": "weird", "execution_count": "7", "outputs": [
        {"type": "stream", "name": "evil", "text": "x" * (sb.MAX_TEXT + 50)},
        *({"type": "image", "png": PNG} for _ in range(sb.MAX_IMAGES + 3)),
    ]}
    cell = make_cell("", reply, 0)
    stream = cell["outputs"][0]
    assert stream["name"] == "stdout" and "50 more characters" in stream["text"]
    assert sum(o["type"] == "image" for o in cell["outputs"]) == sb.MAX_IMAGES
    assert cell["status"] == "error" and cell["execution_count"] is None


def test_summary_gives_the_model_text_not_images() -> None:
    cell = make_cell("plot", {"status": "ok", "execution_count": 2, "outputs": [
        {"type": "stream", "name": "stdout", "text": "ready\n"},
        {"type": "latex", "latex": "\\frac{1}{2}", "text": "1/2"},
        {"type": "image", "png": PNG},
    ]}, 5)
    text = summarize(cell)
    assert "In [2]" in text and "ready" in text and "\\frac{1}{2}" in text
    assert "[figure 1 is shown to the user]" in text and PNG not in text


# --- the pool -----------------------------------------------------------------------


class FakeSandbox:
    started = 0

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on, self.closed, self.running = fail_on, False, 0
        self.max_running = 0

    async def start(self) -> None:
        FakeSandbox.started += 1

    async def execute(self, code: str, timeout: float) -> dict:
        if code == self.fail_on:
            raise SandboxError("boom")
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        await asyncio.sleep(0.01)
        self.running -= 1
        return {"status": "ok", "execution_count": 1, "outputs": [{"type": "text", "text": code}]}

    async def close(self) -> None:
        self.closed = True


def _pool(fail_on: str | None = None, **kwargs) -> tuple[KernelPool, list[FakeSandbox]]:
    boxes: list[FakeSandbox] = []

    def factory() -> FakeSandbox:
        boxes.append(FakeSandbox(fail_on=fail_on))
        return boxes[-1]

    return KernelPool(factory, timeout_s=5, **kwargs), boxes


async def test_pool_reuses_one_sandbox_per_session_and_serialises_runs() -> None:
    pool, boxes = _pool()
    cells = await asyncio.gather(*(pool.execute("s1", f"x{i}") for i in range(3)))
    await pool.execute("s2", "y")
    assert len(boxes) == 2
    assert boxes[0].max_running == 1  # runs in one session never overlap
    assert [c["outputs"][0]["text"] for c in cells] == ["x0", "x1", "x2"]


async def test_pool_evicts_the_least_recently_used_session() -> None:
    pool, boxes = _pool(max_sessions=2)
    await pool.execute("a", "1")
    await pool.execute("b", "1")
    await pool.execute("c", "1")
    await asyncio.sleep(0.02)
    assert boxes[0].closed and not boxes[1].closed
    await pool.close_all()
    assert all(b.closed for b in boxes)


async def test_pool_forgets_a_failed_sandbox() -> None:
    pool, boxes = _pool(fail_on="crash")
    with pytest.raises(SandboxError):
        await pool.execute("s", "crash")
    await pool.execute("s", "fine")
    assert len(boxes) == 2  # a fresh sandbox after the failure


def test_exported_notebooks_start_with_the_sandbox_setup() -> None:
    # web/src/lib/ipynb.ts repeats the executor's SETUP; the two must not drift.
    tree = ast.parse(EXECUTOR.read_text())
    setup = next(node.value.value for node in tree.body
                 if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "SETUP")
    ipynb = (Path(__file__).parents[1] / "web" / "src" / "lib" / "ipynb.ts").read_text()
    literal = re.search(r'const SETUP = (".*?");', ipynb).group(1)
    assert json.loads(literal) == setup.rstrip("\n")


# --- the protocol -------------------------------------------------------------------

# Speaks the executor's protocol; "slow" requests reply late.
FAKE_EXECUTOR = """
import json, sys, time
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    time.sleep(0.4 if request["code"] == "slow" else 0)
    print(json.dumps({"id": request["id"], "status": "ok",
                      "outputs": [{"type": "text", "text": request["code"]}]}), flush=True)
"""


async def test_a_cancelled_run_does_not_shift_later_replies() -> None:
    box = Sandbox([sys.executable, "-c", FAKE_EXECUTOR])
    await box.start()
    try:
        pending = asyncio.create_task(box.execute("slow", 5))
        await asyncio.sleep(0.1)
        pending.cancel()  # as when the user presses Stop mid-run
        with pytest.raises(asyncio.CancelledError):
            await pending
        reply = await box.execute("second", 5)
        assert reply["outputs"][0]["text"] == "second"
    finally:
        await box.close()


# --- the real executor --------------------------------------------------------------

needs_kernel = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("jupyter_client", "ipykernel", "sympy", "matplotlib")),
    reason="needs the notebook extra: uv run --extra notebook pytest",
)


@needs_kernel
async def test_executor_runs_a_persistent_jupyter_kernel() -> None:
    box = Sandbox([sys.executable, str(EXECUTOR)])
    await box.start()
    try:
        assert (await box.execute("x = sp.symbols('x'); area = 21", 30))["status"] == "ok"

        persisted = await box.execute("area * 2", 30)
        assert persisted["outputs"][0]["text"] == "42"

        algebra = make_cell("sp.integrate(x**2, x)", await box.execute("sp.integrate(x**2, x)", 30), 0)
        assert algebra["outputs"][0]["type"] == "latex" and "x^{3}" in algebra["outputs"][0]["latex"]

        plot = make_cell("plot", await box.execute("plt.plot([1, 2, 3]); plt.show()", 30), 0)
        assert any(o["type"] == "image" for o in plot["outputs"])

        failed = await box.execute("1 / 0", 30)
        assert failed["status"] == "error" and failed["outputs"][-1]["ename"] == "ZeroDivisionError"
        assert "\x1b[" not in failed["outputs"][-1]["traceback"]

        slow = await box.execute("import time; time.sleep(30)", 1)
        assert slow["status"] == "timeout"
        assert (await box.execute("area", 30))["outputs"][0]["text"] == "21"  # interrupted, not restarted
    finally:
        await box.close()
