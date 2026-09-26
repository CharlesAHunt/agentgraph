/** OpenRouter's reasoning effort levels, as offered in the UI. */
export const EFFORTS = [
  { id: "none", name: "Off", summary: "Answers straight away; fastest and cheapest" },
  { id: "minimal", name: "Minimal", summary: "A brief think before answering" },
  { id: "low", name: "Low", summary: "Light reasoning" },
  { id: "medium", name: "Medium", summary: "Balanced depth and speed" },
  { id: "high", name: "High", summary: "Thinks longer on hard questions" },
  { id: "xhigh", name: "Maximum", summary: "Deepest reasoning; slowest, most tokens" },
];

export function effortName(id: string): string {
  return EFFORTS.find((e) => e.id === id)?.name ?? id;
}
