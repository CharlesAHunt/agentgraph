import { streamChat } from "./api";
import type { ModelInfo, StreamEvent, Turn, WireMessage } from "./types";

const MODEL_KEY = "lgraph.model";

function storedModel(): string | null {
  try {
    return localStorage.getItem(MODEL_KEY);
  } catch {
    return null;
  }
}

/** Reasoning blobs are provider-specific (some are signed); never replay them to another model. */
function withoutReasoning(history: WireMessage[]): WireMessage[] {
  return history.map(({ reasoning_details: _dropped, ...rest }) => rest);
}

function describeQuery(name: string, args: Record<string, unknown>): string {
  if (typeof args.query === "string") {
    const years = [args.year_from, args.year_to].filter((y) => typeof y === "number");
    return years.length ? `${args.query} (${years.join("–")})` : args.query;
  }
  return name === "list_publications" ? "List the whole corpus" : name;
}

function networkMessage(err: unknown): string {
  const text = err instanceof Error ? err.message : String(err);
  if (/failed to fetch|load failed|networkerror/i.test(text)) {
    return "Can't reach the lgraph server. Start it with `uv run lgraph` and try again.";
  }
  return text;
}

class Chat {
  turns = $state<Turn[]>([]);
  /** Full wire history from the last completed turn; resent with each question. */
  history = $state.raw<WireMessage[]>([]);
  busy = $derived(this.turns.at(-1)?.status === "streaming");
  /** Models the server accepts; empty when it doesn't offer a choice. */
  models = $state.raw<ModelInfo[]>([]);
  defaultModel = $state("");
  /** Selected model id; "" until the model list loads. */
  model = $state("");

  #controller: AbortController | null = null;
  #seq = 0;

  setModels(list: { default: string; models: ModelInfo[] }): void {
    this.models = list.models;
    this.defaultModel = list.default;
    const saved = storedModel();
    this.model = saved && list.models.some((m) => m.id === saved) ? saved : list.default;
  }

  selectModel(id: string): void {
    this.model = id;
    try {
      localStorage.setItem(MODEL_KEY, id);
    } catch {
      /* storage unavailable; the choice lasts for this page only */
    }
  }

  modelName(id: string): string {
    return this.models.find((m) => m.id === id)?.name ?? id;
  }

  ask(question: string): void {
    const q = question.trim();
    if (!q || this.busy) return;
    const model = this.model || this.defaultModel;
    const previous = this.turns.findLast((t) => t.status === "done")?.model;
    const history = previous && previous !== model ? withoutReasoning(this.history) : this.history;
    this.turns.push({
      id: ++this.#seq,
      question: q,
      model,
      text: "",
      reasoning: "",
      searches: [],
      sources: [],
      status: "streaming",
      history,
    });
    const turn = this.turns[this.turns.length - 1];
    void this.#run(turn, [...history, { role: "user", content: q }]);
  }

  stop(): void {
    this.#controller?.abort();
  }

  /** Re-ask the last question after an error or a stop. */
  retry(): void {
    const last = this.turns.at(-1);
    if (!last || last.status === "streaming" || last.status === "done") return;
    this.turns.pop();
    this.ask(last.question);
  }

  reset(): void {
    this.stop();
    this.turns = [];
    this.history = [];
  }

  async #run(turn: Turn, messages: WireMessage[]): Promise<void> {
    const controller = new AbortController();
    this.#controller = controller;
    let pending = "";
    let frame = 0;
    let finished = false;
    const flush = () => {
      frame = 0;
      if (pending) {
        turn.text += pending;
        pending = "";
      }
    };

    const onEvent = (e: StreamEvent) => {
      switch (e.event) {
        case "token":
          pending += e.data.text;
          frame ||= requestAnimationFrame(flush);
          break;
        case "reasoning":
          turn.reasoning += e.data.text;
          break;
        case "tool_start":
          turn.searches.push({
            id: e.data.id,
            tool: e.data.name,
            query: describeQuery(e.data.name, e.data.args ?? {}),
            status: "running",
            count: 0,
          });
          break;
        case "tool_end": {
          const search = turn.searches.find((s) => s.id === e.data.id);
          if (search) {
            search.status = e.data.status === "error" ? "error" : "done";
            search.count = e.data.sources.length;
          }
          turn.sources.push(...e.data.sources);
          break;
        }
        case "done":
          flush();
          finished = true;
          turn.sources = e.data.sources;
          turn.status = "done";
          this.history = e.data.messages;
          break;
        case "error":
          flush();
          finished = true;
          turn.status = "error";
          turn.error = e.data.detail;
          break;
      }
    };

    try {
      await streamChat(messages, turn.model || null, onEvent, controller.signal);
      if (!finished) {
        flush();
        turn.status = "error";
        turn.error = "The connection closed before the answer finished.";
      }
    } catch (err) {
      cancelAnimationFrame(frame);
      flush();
      if (controller.signal.aborted) {
        turn.status = "stopped";
      } else {
        turn.status = "error";
        turn.error = networkMessage(err);
      }
    } finally {
      if (this.#controller === controller) this.#controller = null;
    }
  }
}

export const chat = new Chat();
