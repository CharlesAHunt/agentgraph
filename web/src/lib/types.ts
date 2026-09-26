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

export type TurnStatus = "streaming" | "done" | "error" | "stopped";

export interface Turn {
  id: number;
  question: string;
  text: string;
  reasoning: string;
  searches: Search[];
  sources: Source[];
  status: TurnStatus;
  error?: string;
  /** History this turn was sent with, so a failed turn can be retried. */
  history: WireMessage[];
}

export type StreamEvent =
  | { event: "token"; data: { text: string } }
  | { event: "reasoning"; data: { text: string } }
  | { event: "tool_start"; data: { id: string; name: string; args: Record<string, unknown> } }
  | { event: "tool_end"; data: { id: string; name: string; status: string; sources: Source[] } }
  | { event: "done"; data: { messages: WireMessage[]; sources: Source[] } }
  | { event: "error"; data: { detail: string; status: number; upstream_status: number | null } };

export interface CorpusInfo {
  enabled: boolean;
  papers: number;
}
