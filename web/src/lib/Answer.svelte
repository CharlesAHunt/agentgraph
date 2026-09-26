<script lang="ts">
  import { renderMarkdown } from "./markdown";
  import { cachedDiagram, renderDiagram } from "./mermaid";
  import type { Paper } from "./types";

  let {
    text,
    papers,
    streaming,
    oncite,
  }: { text: string; papers: Paper[]; streaming: boolean; oncite: (n: number) => void } = $props();

  let host: HTMLDivElement;
  const html = $derived(renderMarkdown(text, papers));

  function paint(fig: HTMLElement, svg: string) {
    const canvas = fig.querySelector<HTMLElement>(".diagram-canvas");
    if (!canvas) return;
    canvas.innerHTML = svg;
    fig.dataset.state = "ready";
  }

  // {@html} replaces the DOM on every token; re-attach finished diagrams from cache.
  $effect(() => {
    void html;
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    for (const fig of host.querySelectorAll<HTMLElement>("figure.diagram")) {
      if (fig.dataset.complete !== "true") {
        // An unclosed fence after the stream ends (stop, error, truncation) will never close.
        fig.dataset.state = streaming ? "pending" : "failed";
        continue;
      }
      const source = fig.querySelector(".diagram-src")?.textContent ?? "";
      const hit = cachedDiagram(source, dark);
      if (hit) paint(fig, hit);
      else if (hit === null) fig.dataset.state = "failed";
      else {
        fig.dataset.state = "rendering";
        renderDiagram(source, dark)
          .then((svg) => fig.isConnected && paint(fig, svg))
          .catch(() => {
            if (fig.isConnected) fig.dataset.state = "failed";
          });
      }
    }
  });

  function onclick(event: MouseEvent) {
    const cite = (event.target as HTMLElement).closest<HTMLElement>("button.cite");
    if (cite?.dataset.paper) oncite(Number(cite.dataset.paper));
  }
</script>

<!-- Clicks are delegated from real <button> citation chips, which are keyboard-operable. -->
<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
<div class="answer" class:streaming bind:this={host} {onclick}>{@html html}</div>

<style>
  .answer {
    max-width: 66ch;
  }
  .answer > :global(:first-child) {
    margin-top: 0;
  }
  .answer :global(p) {
    margin: 0 0 1em;
  }
  .answer :global(h2) {
    font-size: 1.22rem;
    font-weight: 600;
    line-height: 1.3;
    margin: 1.9em 0 0.55em;
    padding-top: 0.85em;
    border-top: 1px solid var(--rule);
  }
  .answer :global(h3) {
    font-size: 1.04rem;
    font-weight: 600;
    margin: 1.5em 0 0.4em;
  }
  .answer :global(strong) {
    font-weight: 650;
  }
  .answer :global(em) {
    color: var(--ink-soft);
  }
  .answer :global(ul),
  .answer :global(ol) {
    margin: 0 0 1.1em;
    padding-left: 1.3em;
  }
  .answer :global(li) {
    margin-bottom: 0.45em;
  }
  .answer :global(li::marker) {
    color: var(--accent);
  }
  .answer :global(hr) {
    border: none;
    border-top: 1px solid var(--rule);
    margin: 1.8em 0;
  }
  .answer :global(code) {
    font-family: var(--font-mono);
    font-size: 0.86em;
    background: var(--surface-sunk);
    padding: 0.1em 0.35em;
    border-radius: 3px;
  }
  .answer :global(pre) {
    background: var(--surface-sunk);
    padding: 12px 14px;
    overflow-x: auto;
    border-radius: 4px;
    font-size: 0.85rem;
  }
  .answer :global(pre code) {
    background: none;
    padding: 0;
  }
  .answer :global(table) {
    display: block;
    overflow-x: auto;
    border-collapse: collapse;
    margin: 0 0 1.2em;
    font-size: 0.92rem;
    font-variant-numeric: tabular-nums;
  }
  .answer :global(th),
  .answer :global(td) {
    border-bottom: 1px solid var(--rule);
    padding: 6px 12px 6px 0;
    text-align: left;
  }

  /* The opening blockquote is the summary callout. */
  .answer :global(blockquote) {
    margin: 0 0 1.2em;
    padding: 14px 18px;
    background: var(--surface-raised);
    border-left: 3px solid var(--accent);
    color: var(--ink);
  }
  .answer > :global(blockquote:first-child) {
    font-size: 1.04rem;
    line-height: 1.6;
  }
  .answer :global(blockquote p:last-child) {
    margin-bottom: 0;
  }

  .answer :global(.stats) {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
    margin: 0 0 1.4em;
  }
  .answer :global(.stat) {
    border: 1px solid var(--rule);
    border-bottom: 3px solid var(--accent);
    background: var(--surface);
    padding: 12px 14px 10px;
  }
  .answer :global(.stat-value) {
    font-family: var(--font-display);
    font-size: 1.45rem;
    font-weight: 600;
    line-height: 1.2;
    color: var(--accent-strong);
    font-variant-numeric: tabular-nums;
  }
  .answer :global(.stat-label) {
    margin-top: 4px;
    font-size: 0.78rem;
    line-height: 1.35;
    color: var(--ink-soft);
  }

  .answer :global(figure.diagram) {
    margin: 0 0 1.4em;
    padding: 16px;
    border: 1px solid var(--rule);
    background: var(--surface-raised);
    overflow-x: auto;
  }
  .answer :global(.diagram-canvas svg) {
    display: block;
    max-width: none;
    height: auto;
    margin-inline: auto;
  }
  .answer :global(.diagram-src) {
    display: none;
    margin: 0;
    background: none;
    padding: 0;
    white-space: pre-wrap;
  }
  .answer :global(figure.diagram[data-state="pending"]),
  .answer :global(figure.diagram[data-state="rendering"]) {
    min-height: 88px;
    display: grid;
    place-items: center;
  }
  .answer :global(figure.diagram[data-state="pending"] .diagram-canvas::before),
  .answer :global(figure.diagram[data-state="rendering"] .diagram-canvas::before) {
    content: "Drawing diagram…";
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--ink-faint);
    letter-spacing: 0.04em;
  }
  .answer :global(figure.diagram[data-state="failed"] .diagram-src) {
    display: block;
  }

  .answer :global(.math-display) {
    margin: 0 0 1.1em;
  }
  .answer :global(.katex) {
    font-size: 1.08em;
  }
  .answer :global(.stat-label .katex) {
    font-size: 1em;
  }

  .answer :global(button.cite) {
    display: inline;
    padding: 0 1px;
    border: none;
    background: none;
    font: inherit;
    color: var(--ink);
    text-decoration: underline;
    text-decoration-color: var(--rule-strong);
    text-underline-offset: 3px;
    border-radius: 2px;
  }
  .answer :global(button.cite:hover) {
    text-decoration-color: var(--accent);
    color: var(--accent-strong);
  }
  .answer :global(.cite-n) {
    display: inline-block;
    margin-left: 3px;
    min-width: 1.35em;
    padding: 0 4px;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    line-height: 1.5;
    text-align: center;
    vertical-align: 0.15em;
    border-radius: 3px;
    background: var(--accent-soft);
    color: var(--accent-strong);
    text-decoration: none;
  }

  .answer.streaming > :global(:last-child:not(ul, ol, figure, .stats, pre))::after,
  .answer.streaming > :global(:is(ul, ol):last-child > li:last-child)::after {
    content: "";
    display: inline-block;
    width: 0.55em;
    height: 1.05em;
    margin-left: 2px;
    vertical-align: -0.15em;
    background: var(--accent);
    animation: caret 1s steps(2, jump-none) infinite;
  }
  @keyframes caret {
    50% {
      opacity: 0;
    }
  }
</style>
