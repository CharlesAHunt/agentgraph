"""Per-conversation Python sandboxes: one container running ``executor.py`` each.

The server talks to each sandbox over the container's stdin/stdout, one JSON
message per line, so the container needs no network at all. Every command is
an argv list run without a shell; container names are generated here.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import shlex
import shutil
import sys
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..text import truncate

logger = logging.getLogger(__name__)

LABEL = "lgraph=notebook"
MAX_SESSIONS = 4
IDLE_S = 15 * 60
REAP_EVERY_S = 60
START_TIMEOUT_S = 120
# Beyond the cell timeout: room for the executor's own interrupt and kernel restart.
WATCHDOG_MARGIN_S = 60
STREAM_LIMIT = 32 * 1024 * 1024

MAX_TEXT = 20_000
MAX_IMAGES = 6
MAX_IMAGE_BYTES = 2 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
STATUSES = frozenset({"ok", "error", "timeout"})

EXECUTOR = Path(__file__).with_name("executor.py")


class SandboxError(RuntimeError):
    """The sandbox could not run the code. Messages are safe to show to users."""


# --- commands ----------------------------------------------------------------


def detect_runtime() -> list[str] | None:
    """The container CLI: the host's podman from inside a distrobox, else podman or docker."""
    in_container = bool(os.environ.get("CONTAINER_ID")) or Path("/run/.containerenv").exists()
    if in_container and shutil.which("distrobox-host-exec"):
        return ["distrobox-host-exec", "podman"]
    for cli in ("podman", "docker"):
        if shutil.which(cli):
            return [cli]
    return None


def container_command(runtime: list[str], image: str, name: str) -> list[str]:
    return [
        *runtime, "run", "--rm", "-i", "--pull", "never",
        "--name", name, "--label", LABEL,
        "--network", "none",
        "--memory", "1g", "--memory-swap", "1g", "--cpus", "1", "--pids-limit", "256",
        "--read-only", "--tmpfs", "/tmp:rw,size=256m,mode=1777",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        image,
    ]


async def _run(argv: list[str], timeout: float = 30) -> tuple[int, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except (OSError, TimeoutError) as exc:
        return 1, str(exc)
    return proc.returncode or 0, out.decode(errors="replace")


# --- one sandbox ---------------------------------------------------------------


class Sandbox:
    """One executor process: a container, or a plain subprocess in unsafe-local mode."""

    def __init__(self, argv: list[str], *, remove: list[str] | None = None) -> None:
        self._argv = argv
        self._remove = remove
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr: deque[str] = deque(maxlen=20)
        self._stderr_task: asyncio.Task | None = None
        self._seq = 0

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=STREAM_LIMIT,
        )
        self._stderr_task = asyncio.create_task(self._collect_stderr())
        try:
            line = await asyncio.wait_for(self._proc.stdout.readline(), START_TIMEOUT_S)
            ready = bool(line) and json.loads(line).get("ready") is True
        except (TimeoutError, ValueError):
            ready = False
        if not ready:
            logger.error("Python sandbox failed to start: %s", " | ".join(self._stderr) or "no output")
            await self.close()
            raise SandboxError("the Python sandbox did not start; see the server log")

    async def execute(self, code: str, timeout: float) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
        self._seq += 1
        request_id = self._seq
        request = json.dumps({"id": request_id, "code": code, "timeout": timeout}) + "\n"
        try:
            self._proc.stdin.write(request.encode())
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            return await self._exited()
        # The executor enforces the timeout itself; this deadline catches a wedged container.
        deadline = time.monotonic() + timeout + WATCHDOG_MARGIN_S
        while True:
            try:
                line = await asyncio.wait_for(self._proc.stdout.readline(), deadline - time.monotonic())
            except TimeoutError:
                await self.close()
                raise SandboxError("the Python kernel stopped responding and was shut down") from None
            except ValueError:
                line = b""
            if not line:
                return await self._exited()
            try:
                reply = json.loads(line)
            except ValueError:
                await self.close()
                raise SandboxError("the Python kernel sent an unreadable reply and was shut down") from None
            # A reply to an earlier request whose caller was cancelled (e.g. the user pressed Stop).
            if reply.get("id") == request_id:
                return reply

    async def _exited(self) -> dict[str, Any]:
        logger.warning("Python sandbox exited: %s", " | ".join(self._stderr) or "no output")
        await self.close()
        raise SandboxError("the Python kernel exited, possibly after running out of memory")

    async def close(self) -> None:
        if self._remove:
            await _run(self._remove)
        if self._proc and self._proc.returncode is None:
            self._proc.kill()
            await self._proc.wait()
        if self._stderr_task:
            self._stderr_task.cancel()

    async def _collect_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        async for raw in self._proc.stderr:
            self._stderr.append(raw.decode(errors="replace").rstrip())


# --- output limits ---------------------------------------------------------------


def _text(value: Any) -> str:
    return truncate(str(value or ""), MAX_TEXT)


def _png(value: Any) -> str | None:
    """Re-encoded base64 if ``value`` really is a PNG within the size limit."""
    try:
        raw = base64.b64decode(str(value), validate=True)
    except (binascii.Error, ValueError):
        return None
    if not raw.startswith(PNG_SIGNATURE) or len(raw) > MAX_IMAGE_BYTES:
        return None
    return base64.b64encode(raw).decode("ascii")


def make_cell(code: str, reply: dict[str, Any], duration_ms: int) -> dict[str, Any]:
    """The executor's reply as a cell for the model and UI, bounded and type-checked."""
    outputs: list[dict[str, Any]] = []
    images = 0
    for out in reply.get("outputs") or []:
        kind = out.get("type") if isinstance(out, dict) else None
        if kind == "stream":
            name = "stderr" if out.get("name") == "stderr" else "stdout"
            outputs.append({"type": "stream", "name": name, "text": _text(out.get("text"))})
        elif kind == "text":
            outputs.append({"type": "text", "text": _text(out.get("text"))})
        elif kind == "latex":
            outputs.append({"type": "latex", "latex": _text(out.get("latex")), "text": _text(out.get("text"))})
        elif kind == "image" and images < MAX_IMAGES and (png := _png(out.get("png"))):
            images += 1
            outputs.append({"type": "image", "png": png, "text": _text(out.get("text"))})
        elif kind == "error":
            outputs.append({"type": "error", "ename": _text(out.get("ename")),
                            "evalue": _text(out.get("evalue")), "traceback": _text(out.get("traceback"))})
    count = reply.get("execution_count")
    status = reply.get("status")
    return {
        "id": uuid.uuid4().hex[:12],
        "code": code,
        "status": status if status in STATUSES else "error",
        "execution_count": count if isinstance(count, int) else None,
        "outputs": outputs,
        "duration_ms": duration_ms,
    }


# --- the pool ----------------------------------------------------------------------


@dataclass
class _Entry:
    ready: asyncio.Task
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = field(default_factory=time.monotonic)


class KernelPool:
    """Sandboxes by conversation session, started on first use and reaped when idle."""

    def __init__(
        self,
        factory: Callable[[], Sandbox],
        *,
        timeout_s: float,
        max_sessions: int = MAX_SESSIONS,
        idle_s: float = IDLE_S,
    ) -> None:
        self._factory = factory
        self.timeout_s = timeout_s
        self._max = max_sessions
        self._idle_s = idle_s
        self._sessions: dict[str, _Entry] = {}
        self._background: set[asyncio.Task] = set()

    async def execute(self, session: str, code: str) -> dict[str, Any]:
        entry = self._entry(session)
        async with entry.lock:
            try:
                sandbox = await entry.ready
                started = time.monotonic()
                reply = await sandbox.execute(code, self.timeout_s)
            except SandboxError:
                self._forget(session, entry)
                raise
            finally:
                entry.last_used = time.monotonic()
        return make_cell(code, reply, int((time.monotonic() - started) * 1000))

    async def close(self, session: str) -> None:
        entry = self._sessions.pop(session, None)
        if entry:
            await self._shutdown(entry)

    async def close_all(self) -> None:
        entries, self._sessions = list(self._sessions.values()), {}
        await asyncio.gather(*(self._shutdown(e) for e in entries), return_exceptions=True)

    async def reap_forever(self) -> None:
        while True:
            await asyncio.sleep(REAP_EVERY_S)
            cutoff = time.monotonic() - self._idle_s
            for session, entry in list(self._sessions.items()):
                if entry.last_used < cutoff and not entry.lock.locked():
                    await self.close(session)

    def _entry(self, session: str) -> _Entry:
        entry = self._sessions.get(session)
        if entry is None:
            if len(self._sessions) >= self._max:
                self._evict()
            entry = _Entry(asyncio.create_task(self._start()))
            self._sessions[session] = entry
        return entry

    async def _start(self) -> Sandbox:
        sandbox = self._factory()
        await sandbox.start()
        return sandbox

    def _evict(self) -> None:
        idle = [(e.last_used, s) for s, e in self._sessions.items() if not e.lock.locked()]
        if not idle:
            raise SandboxError("every Python kernel is busy; try again in a moment")
        _, session = min(idle)
        self._spawn(self._shutdown(self._sessions.pop(session)))

    def _forget(self, session: str, entry: _Entry) -> None:
        if self._sessions.get(session) is entry:
            del self._sessions[session]

    async def _shutdown(self, entry: _Entry) -> None:
        try:
            sandbox = await entry.ready
        except (SandboxError, asyncio.CancelledError):
            return
        await sandbox.close()

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)


# --- construction from settings ---------------------------------------------------------


async def open_kernel_pool(settings: Settings) -> KernelPool | None:
    """The pool for ``LGRAPH_PYTHON``, or None when it is off or cannot run."""
    if settings.python == "off":
        return None
    if settings.python == "unsafe-local":
        logger.warning(
            "LGRAPH_PYTHON=unsafe-local: model-written code runs on this machine WITHOUT isolation. "
            "Use it only for development."
        )
        return KernelPool(lambda: Sandbox([sys.executable, str(EXECUTOR)]), timeout_s=settings.python_timeout_s)

    runtime = shlex.split(settings.python_runtime) if settings.python_runtime else detect_runtime()
    if not runtime:
        logger.error("LGRAPH_PYTHON=container but no podman or docker was found; run_python is disabled.")
        return None
    code, out = await _run([*runtime, "image", "inspect", settings.python_image])
    if code != 0:
        logger.error(
            "Sandbox image %r not found (%s); run_python is disabled. Build it with: %s build -t %s src/lgraph/notebook",
            settings.python_image, out.strip()[:200], " ".join(runtime), settings.python_image,
        )
        return None
    # Containers left behind by a previous server process.
    code, out = await _run([*runtime, "ps", "-aq", "--filter", f"label={LABEL}"])
    if code == 0 and out.split():
        await _run([*runtime, "rm", "-f", *out.split()])

    def factory() -> Sandbox:
        name = f"lgraph-py-{uuid.uuid4().hex[:12]}"
        return Sandbox(
            container_command(runtime, settings.python_image, name),
            remove=[*runtime, "rm", "-f", name],
        )

    logger.info("run_python enabled: %s, image %s", " ".join(runtime), settings.python_image)
    return KernelPool(factory, timeout_s=settings.python_timeout_s)
