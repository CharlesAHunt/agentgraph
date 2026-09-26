import { streamChat } from "./api";
import { EFFORTS } from "./efforts";
import { CUSTOM_ROLE, DEFAULT_ROLE, MAX_ROLE_LENGTH, ROLES } from "./roles";
import type { ModelInfo, StreamEvent, Turn, WireMessage } from "./types";

const MODEL_KEY = "lgraph.model";
const ROLE_KEY = "lgraph.role";
const EFFORT_KEY = "lgraph.effort";

function load(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function save(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage unavailable; the choice lasts for this page only */
  }
}

function storedRole(): { id: string; custom: string } {
  try {
    const { id, custom } = JSON.parse(load(ROLE_KEY) ?? "{}");
    const known = id === CUSTOM_ROLE || ROLES.some((r) => r.id === id);
    return { id: known ? id : DEFAULT_ROLE, custom: typeof custom === "string" ? custom.slice(0, MAX_ROLE_LENGTH) : "" };
  } catch {
    return { id: DEFAULT_ROLE, custom: "" };
  }
}

const savedRole = storedRole();
const savedEffort = EFFORTS.find((e) => e.id === load(EFFORT_KEY))?.id ?? "";

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
  roleId = $state(savedRole.id);
  customRole = $state(savedRole.custom);
  /** The role as it will be sent with the next question. */
  role = $derived.by(() => {
    if (this.roleId === CUSTOM_ROLE) return { name: "Custom role", instructions: this.customRole.trim() };
    const preset = ROLES.find((r) => r.id === this.roleId) ?? ROLES[0];
    return { name: preset.name, instructions: preset.instructions };
  });

  defaultEffort = $state("medium");
  /** Chosen reasoning effort; "" means the server default. */
  effort = $state(savedEffort);
  /** Whether the selected model takes a reasoning effort at all. */
  reasons = $derived(this.models.find((m) => m.id === (this.model || this.defaultModel))?.reasoning ?? true);

  #controller: AbortController | null = null;
  #seq = 0;

  setModels(list: { default: string; default_effort?: string; models: ModelInfo[] }): void {
    this.models = list.models;
    this.defaultModel = list.default;
    if (list.default_effort) this.defaultEffort = list.default_effort;
    const saved = load(MODEL_KEY);
    this.model = saved && list.models.some((m) => m.id === saved) ? saved : list.default;
  }

  setEffort(id: string): void {
    this.effort = id;
    save(EFFORT_KEY, id);
  }

  selectModel(id: string): void {
    this.model = id;
    save(MODEL_KEY, id);
  }

  setRole(id: string, custom = this.customRole): void {
    this.roleId = id;
    this.customRole = custom.slice(0, MAX_ROLE_LENGTH);
    save(ROLE_KEY, JSON.stringify({ id: this.roleId, custom: this.customRole }));
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
      role: { ...this.role },
      effort: this.reasons && this.effort ? this.effort : null,
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
      await streamChat(
        messages,
        { model: turn.model, instructions: turn.role.instructions, effort: turn.effort },
        onEvent,
        controller.signal,
      );
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
