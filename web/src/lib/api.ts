import type { Cell, CorpusInfo, ModelInfo, StreamEvent, UsageReport, WireMessage } from "./types";

/** A JSON request whose failure the caller shows to the user. */
async function request<T>(url: string, init?: RequestInit): Promise<{ data: T } | { error: string }> {
  try {
    const res = await fetch(url, init);
    if (!res.ok) return { error: await errorDetail(res) };
    return { data: await res.json() };
  } catch {
    return { error: "Can't reach the lgraph server." };
  }
}

/** A JSON GET whose failure just hides the feature; null on any error. */
async function getJson(url: string): Promise<any | null> {
  try {
    const res = await fetch(url);
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

/** Run code in a conversation's kernel, e.g. a cell the user edited. */
export function executeCell(session: string, code: string) {
  return request<Cell>(`/sessions/${encodeURIComponent(session)}/execute`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ code }),
  });
}

/** Shut a conversation's kernel down; best effort. */
export function endSession(session: string): void {
  void fetch(`/sessions/${encodeURIComponent(session)}`, { method: "DELETE", keepalive: true }).catch(() => {});
}

export function fetchUsage() {
  return request<UsageReport>("/usage");
}

export async function fetchModels(): Promise<{ default: string; default_effort?: string; models: ModelInfo[] } | null> {
  const body = await getJson("/models");
  return typeof body?.default === "string" && Array.isArray(body.models) ? body : null;
}

export async function fetchCorpus(): Promise<CorpusInfo | null> {
  const body = await getJson("/papers?count_only=true");
  return body ? { enabled: Boolean(body.enabled), papers: Number(body.count) || 0 } : null;
}

/**
 * POST a conversation to /chat/stream and call `onEvent` for each server-sent
 * event. Resolves when the stream ends; rejects on HTTP errors or abort.
 */
export async function streamChat(
  messages: WireMessage[],
  options: { model?: string; instructions?: string; effort?: string | null; session?: string },
  onEvent: (e: StreamEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch("/chat/stream", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages,
      model: options.model || undefined,
      instructions: options.instructions || undefined,
      effort: options.effort || undefined,
      session: options.session || undefined,
    }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(await errorDetail(res));

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let end: number;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const parsed = parseFrame(frame);
      if (parsed) onEvent(parsed);
    }
  }
}

function parseFrame(frame: string): StreamEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return null;
  try {
    return { event, data: JSON.parse(data.join("\n")) } as StreamEvent;
  } catch {
    return null;
  }
}

async function errorDetail(res: Response): Promise<string> {
  const fallback = res.status === 404
    ? "The server is older than this page. Restart it with `uv run lgraph`."
    : `The server answered ${res.status}.`;
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg?: string }) => d.msg).join("; ");
  } catch {
    /* non-JSON body */
  }
  return fallback;
}
