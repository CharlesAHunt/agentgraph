/** A message in the service's wire format. Sent back unmodified each turn. */
export interface WireMessage {
  role: string;
  content?: string | null;
  reasoning_details?: unknown;
  tool_calls?: unknown[];
  tool_call_id?: string;
  name?: string;
}

/** One retrieved excerpt, as returned by search_publications. */
export interface Source {
  n?: number;
  title: string;
  authors: string[];
  year: number | null;
  doi?: string | null;
  arxiv_id?: string | null;
  section: string;
  page: number;
  chunk_id: string;
}

/** Sources grouped by paper, numbered in the order they were first retrieved. */
export interface Paper {
  n: number;
  key: string;
  title: string;
  authors: string[];
  year: number | null;
  doi?: string | null;
  arxiv_id?: string | null;
  cites: { section: string; page: number }[];
}

export interface Search {
  id: string;
  tool: string;
  query: string;
  status: "running" | "done" | "error";
  count: number;
}

/** The notebook tool's name, as the server reports it in tool events. */
export const PYTHON_TOOL = "run_python";

export type CellOutput =
  | { type: "stream"; name: "stdout" | "stderr"; text: string }
  | { type: "text"; text: string }
  | { type: "latex"; latex: string; text: string }
  | { type: "image"; png: string; text: string }
  | { type: "error"; ename: string; evalue: string; traceback: string };

/** One run_python execution, as returned by the server. */
export interface Cell {
  id: string;
  code: string;
  status: "ok" | "error" | "timeout";
  execution_count: number | null;
  outputs: CellOutput[];
  duration_ms: number;
  /** Re-run by the user after editing. */
  edited?: boolean;
}

export type TurnStatus = "streaming" | "done" | "error" | "stopped";

export interface Turn {
  id: number;
  question: string;
  /** Model id this turn was sent to. */
  model: string;
  /** Role this turn was asked in, as sent to the server. */
  role: { name: string; instructions: string };
  /** Reasoning effort sent with this turn; null means the server default. */
  effort: string | null;
  text: string;
  reasoning: string;
  searches: Search[];
  sources: Source[];
  cells: Cell[];
  status: TurnStatus;
  error?: string;
}

export type StreamEvent =
  | { event: "token"; data: { text: string } }
  | { event: "reasoning"; data: { text: string } }
  | { event: "tool_start"; data: { id: string; name: string; args: Record<string, unknown> } }
  | { event: "tool_end"; data: { id: string; name: string; status: string; sources: Source[]; cell?: Cell | null } }
  | { event: "done"; data: { messages: WireMessage[]; sources: Source[]; cells?: Cell[] } }
  | { event: "error"; data: { detail: string; status: number; upstream_status: number | null } };

/** A chat model from GET /models. Prices are USD per million tokens; null means variable. */
export interface ModelInfo {
  id: string;
  name: string;
  context_length: number | null;
  prompt_price: number | null;
  completion_price: number | null;
  /** Whether the model accepts a reasoning effort. */
  reasoning: boolean;
}

/** GET /usage, all amounts in USD. */
export interface UsageReport {
  key: {
    usage: number;
    usage_daily: number;
    usage_weekly: number;
    usage_monthly: number;
    limit: number | null;
    limit_remaining: number | null;
    limit_reset: string | null;
    is_free_tier: boolean;
  };
  credits: { total: number; used: number; remaining: number } | null;
  credits_note: string | null;
}

export interface CorpusInfo {
  enabled: boolean;
  papers: number;
}
