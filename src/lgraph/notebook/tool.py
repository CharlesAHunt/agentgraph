"""The ``run_python`` agent tool over a conversation's sandboxed kernel."""

from __future__ import annotations

from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool

from ..agent import TurnContext
from ..text import truncate
from .sandbox import KernelPool, SandboxError

TOOL_NAME = "run_python"
MAX_SUMMARY = 6_000
NO_SESSION = "The Python tool is unavailable: this request has no conversation session."


def summarize(cell: dict[str, Any]) -> str:
    """What the model sees: text results only; figures go to the user."""
    count = cell.get("execution_count")
    parts = [f"status: {cell['status']}" + (f" (In [{count}])" if count else "")]
    figures = 0
    for out in cell["outputs"]:
        kind = out["type"]
        if kind == "stream":
            parts.append(f"{out['name']}:\n{out['text'].rstrip()}")
        elif kind == "latex":
            parts.append(f"result (LaTeX): {out['latex'].strip()}")
        elif kind == "text":
            parts.append(f"result: {out['text'].strip()}")
        elif kind == "image":
            figures += 1
            parts.append(f"[figure {figures} is shown to the user]")
        elif kind == "error":
            parts.append(f"error: {out['ename']}: {out['evalue']}\n{out['traceback'][-2000:]}".rstrip())
    if len(parts) == 1:
        parts.append("(no output)")
    return truncate("\n".join(parts), MAX_SUMMARY)


def make_python_tool(pool: KernelPool) -> BaseTool:
    async def run_python(code: str, runtime: ToolRuntime[TurnContext]) -> tuple[str, dict[str, Any]]:
        """Run Python code in this conversation's Jupyter kernel and return its output.

        Variables persist between calls for the whole conversation. numpy (np),
        sympy (sp) and matplotlib.pyplot (plt) are already imported; scipy and
        pandas are installed. Use it to check derivations, solve or simplify
        equations, evaluate formulas with numbers and plot results (figures are
        shown to the user). There is no internet or file access and no access
        to the paper corpus. Print or return only what matters.
        """
        session = getattr(runtime.context, "session", None)
        if not session:
            return NO_SESSION, {}
        try:
            cell = await pool.execute(session, code)
        except SandboxError as exc:
            return f"The Python kernel is unavailable: {exc}.", {}
        return summarize(cell), cell

    return StructuredTool.from_function(
        coroutine=run_python, name=TOOL_NAME, response_format="content_and_artifact"
    )
