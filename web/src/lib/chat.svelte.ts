import { streamChat } from "./api";
import type { StreamEvent, Turn, WireMessage } from "./types";

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

  #controller: AbortController | null = null;
  #seq = 0;

  ask(question: string): void {
    const q = question.trim();
    if (!q || this.busy) return;
    this.turns.push({
      id: ++this.#seq,
      question: q,
      text: "",
      reasoning: "",
      searches: [],
      sources: [],
      status: "streaming",
      history: this.history,
    });
    const turn = this.turns[this.turns.length - 1];
    void this.#run(turn, [...this.history, { role: "user", content: q }]);
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
      await streamChat(messages, onEvent, controller.signal);
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
