"""Jupyter kernel driver that runs inside the Python sandbox.

Reads one JSON request per line on stdin (``{"id", "code", "timeout"}``),
runs the code in a persistent ipykernel and writes one JSON reply per line on
stdout. Needs only the standard library and ``jupyter_client``, so the file
can be copied into the sandbox image on its own.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from queue import Empty
from typing import Any, TextIO

from jupyter_client import KernelManager

# Mirrored in web/src/lib/ipynb.ts for exported notebooks; a test keeps them equal.
SETUP = "%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\n"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
MAX_STREAM_CHARS = 100_000
INTERRUPT_GRACE_S = 5.0


def error_output(ename: str, evalue: str, traceback: str = "") -> dict[str, Any]:
    return {"type": "error", "ename": ename, "evalue": evalue, "traceback": traceback}


def rich(data: dict[str, Any]) -> dict[str, Any]:
    """One display output, keeping the richest form the UI can show."""
    text = str(data.get("text/plain", ""))
    if "image/png" in data:
        return {"type": "image", "png": str(data["image/png"]).strip(), "text": text}
    if "text/latex" in data:
        return {"type": "latex", "latex": str(data["text/latex"]), "text": text}
    return {"type": "text", "text": text}


def add_stream(outputs: list[dict[str, Any]], name: str, text: str) -> None:
    last = outputs[-1] if outputs else None
    if last and last["type"] == "stream" and last["name"] == name:
        last["text"] = (last["text"] + text)[:MAX_STREAM_CHARS]
    else:
        outputs.append({"type": "stream", "name": name, "text": text[:MAX_STREAM_CHARS]})


class Kernel:
    def __init__(self) -> None:
        self.km = KernelManager(kernel_name="python3")
        # The kernel's own stdout/stderr would otherwise land in our protocol stream.
        self.km.start_kernel(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.kc = self.km.client()
        self.kc.start_channels()
        self._ready()

    def _ready(self) -> None:
        self.kc.wait_for_ready(timeout=60)
        self.run(SETUP, 60)

    def run(self, code: str, timeout: float) -> dict[str, Any]:
        msg_id = self.kc.execute(code, store_history=True, allow_stdin=False)
        outputs: list[dict[str, Any]] = []
        status, count = "ok", None
        deadline = time.monotonic() + timeout
        interrupted = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if interrupted:
                    # The interrupt did not take; start over with a fresh kernel.
                    self.km.restart_kernel(now=True)
                    self._ready()
                    outputs.append(error_output(
                        "Timeout", "The kernel did not stop and was restarted; variables were lost."))
                    return {"status": "timeout", "execution_count": count, "outputs": outputs}
                self.km.interrupt_kernel()
                interrupted, status = True, "timeout"
                deadline = time.monotonic() + INTERRUPT_GRACE_S
                continue
            try:
                msg = self.kc.get_iopub_msg(timeout=min(remaining, 1.0))
            except Empty:
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            kind, content = msg["msg_type"], msg["content"]
            if kind == "status" and content.get("execution_state") == "idle":
                break
            if kind == "execute_input":
                count = content.get("execution_count")
            elif kind == "stream":
                add_stream(outputs, content.get("name", "stdout"), content.get("text", ""))
            elif kind in ("execute_result", "display_data"):
                outputs.append(rich(content.get("data", {})))
            elif kind == "error":
                status = "timeout" if interrupted else "error"
                outputs.append(error_output(
                    content.get("ename", ""),
                    content.get("evalue", ""),
                    ANSI.sub("", "\n".join(content.get("traceback", []))),
                ))
        if interrupted:
            outputs.append(error_output("Timeout", f"Stopped after {timeout:g} s."))
        return {"status": status, "execution_count": count, "outputs": outputs}

    def shutdown(self) -> None:
        self.kc.stop_channels()
        self.km.shutdown_kernel(now=True)


def reply(out: TextIO, message: dict[str, Any]) -> None:
    out.write(json.dumps(message) + "\n")
    out.flush()


def main() -> None:
    # Keep the real stdout for the protocol; anything else printing to fd 1 goes to stderr.
    protocol = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    kernel = Kernel()
    reply(protocol, {"ready": True})
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        try:
            result = kernel.run(str(request["code"]), float(request.get("timeout", 30)))
        except Exception as exc:  # noqa: BLE001 - report it and keep serving
            result = {"status": "error", "execution_count": None,
                      "outputs": [error_output(type(exc).__name__, str(exc))]}
        reply(protocol, {"id": request.get("id"), **result})
    kernel.shutdown()


if __name__ == "__main__":
    main()
