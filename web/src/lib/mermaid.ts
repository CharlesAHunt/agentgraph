import type { Mermaid } from "mermaid";

let loader: Promise<Mermaid> | null = null;
let seq = 0;
/** svg string, or null when the source failed to parse. */
const done = new Map<string, string | null>();
const inflight = new Map<string, Promise<string>>();

const key = (source: string, dark: boolean) => `${dark ? "d" : "l"}:${source}`;

export function cachedDiagram(source: string, dark: boolean): string | null | undefined {
  return done.get(key(source, dark));
}

/** Render a mermaid diagram once per (source, theme). Loads mermaid on first use. */
export function renderDiagram(source: string, dark: boolean): Promise<string> {
  const k = key(source, dark);
  const pending = inflight.get(k);
  if (pending) return pending;
  const job = (async () => {
    loader ??= import("mermaid").then((m) => m.default);
    const mermaid = await loader;
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme: "base",
      fontFamily: '"Public Sans", "Helvetica Neue", Arial, sans-serif',
      themeVariables: themeVariables(),
      // Natural size keeps labels legible; the figure scrolls if a chart is wider than the column.
      flowchart: { htmlLabels: false, curve: "basis", useMaxWidth: false },
    });
    try {
      const { svg } = await mermaid.render(`lgraph-diagram-${++seq}`, source);
      done.set(k, svg);
      return svg;
    } catch (err) {
      done.set(k, null);
      throw err;
    } finally {
      inflight.delete(k);
    }
  })();
  inflight.set(k, job);
  return job;
}

function themeVariables(): Record<string, string> {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string) => css.getPropertyValue(name).trim();
  return {
    background: v("--surface"),
    primaryColor: v("--surface-raised"),
    primaryBorderColor: v("--accent"),
    primaryTextColor: v("--ink"),
    secondaryColor: v("--accent-soft"),
    tertiaryColor: v("--bg"),
    lineColor: v("--ink-faint"),
    textColor: v("--ink"),
    edgeLabelBackground: v("--surface"),
    clusterBkg: v("--bg"),
    clusterBorder: v("--rule"),
    fontSize: "14px",
  };
}
