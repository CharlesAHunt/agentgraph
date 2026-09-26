import type { CorpusInfo, ModelInfo, StreamEvent, UsageReport, WireMessage } from "./types";

export async function fetchUsage(): Promise<{ report: UsageReport } | { error: string }> {
  try {
    const res = await fetch("/usage");
    if (!res.ok) return { error: await errorDetail(res) };
    return { report: await res.json() };
  } catch {
    return { error: "Can't reach the lgraph server." };
  }
}

export async function fetchModels(): Promise<{ default: string; models: ModelInfo[] } | null> {
  try {
    const res = await fetch("/models");
    if (!res.ok) return null;
    const body = await res.json();
    return typeof body.default === "string" && Array.isArray(body.models) ? body : null;
  } catch {
    return null;
  }
}

export async function fetchCorpus(): Promise<CorpusInfo | null> {
  try {
    const res = await fetch("/papers");
    if (!res.ok) return null;
    const body = await res.json();
    return { enabled: Boolean(body.enabled), papers: Array.isArray(body.papers) ? body.papers.length : 0 };
  } catch {
    return null;
  }
}

/**
 * POST a conversation to /chat/stream and call `onEvent` for each server-sent
 * event. Resolves when the stream ends; rejects on HTTP errors or abort.
 */
export async function streamChat(
  messages: WireMessage[],
  model: string | null,
  onEvent: (e: StreamEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch("/chat/stream", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(model ? { messages, model } : { messages }),
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
