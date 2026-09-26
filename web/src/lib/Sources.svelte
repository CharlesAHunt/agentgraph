<script lang="ts">
  import { plural } from "./format";
  import { paperLink, shortAuthors } from "./sources";
  import type { Paper } from "./types";

  let {
    papers,
    turnId,
    flash,
    live,
  }: { papers: Paper[]; turnId: number; flash: number | null; live: boolean } = $props();

  const COLLAPSED = 6;
  let expanded = $state(false);
  const shown = $derived(expanded ? papers : papers.slice(0, COLLAPSED));

  // While the answer streams, a one-line summary keeps the text being written in view.
  let opened = $state(false);
  const summaryOnly = $derived(live && !opened && flash === null);
  const lead = (p: Paper) => `${p.authors[0]?.trim().split(/\s+/).pop() ?? p.title} ${p.year ?? ""}`.trim();
  const leads = $derived(
    papers.slice(0, 3).map(lead).join(", ") + (papers.length > 3 ? ` +${papers.length - 3}` : ""),
  );

  // A citation click on a hidden paper expands the list so it can be scrolled to.
  $effect(() => {
    if (flash !== null && flash > COLLAPSED) expanded = true;
  });
</script>

<section class="sources" aria-label="Sources">
  {#if summaryOnly}
    <button type="button" class="summary" onclick={() => (opened = true)} title="Show the sources">
      <span class="section-heading">Sources <span>{plural(papers.length, "paper")}</span></span>
      <span class="leads">{leads}</span>
      <span class="show">Show</span>
    </button>
  {:else}
    <h3 class="section-heading">Sources <span>{plural(papers.length, "paper")}</span></h3>
    <ol>
      {#each shown as p (p.key)}
        {@const link = paperLink(p)}
        <li id="t{turnId}-p{p.n}" class:flash={flash === p.n}>
          <span class="num">{String(p.n).padStart(2, "0")}</span>
          <div class="body">
            <div class="title">{p.title}</div>
            <div class="meta">{shortAuthors(p.authors)} · {p.year ?? "n.d."}</div>
            <div class="row">
              {#if link}
                <a href={link.href} target="_blank" rel="noopener noreferrer">{link.label} ↗</a>
              {/if}
              {#each p.cites as c (`${c.page}:${c.section}`)}
                <span class="cite">§ {c.section}, p. {c.page}</span>
              {/each}
            </div>
          </div>
        </li>
      {/each}
    </ol>
    {#if papers.length > COLLAPSED}
      <button type="button" class="more" onclick={() => (expanded = !expanded)}>
        {expanded ? "Show fewer" : `Show all ${papers.length} papers`}
      </button>
    {/if}
  {/if}
</section>

<style>
  .sources {
    margin-top: 28px;
    padding-top: 18px;
    border-top: 1px solid var(--rule);
  }
  h3 {
    margin-bottom: 6px;
  }
  .summary {
    display: flex;
    align-items: baseline;
    gap: 10px;
    width: 100%;
    padding: 0;
    border: none;
    background: none;
    text-align: left;
    color: inherit;
  }
  .leads {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 0.84rem;
    color: var(--ink-soft);
  }
  .show {
    flex: none;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--accent-strong);
  }
  .summary:hover .show {
    text-decoration: underline;
  }
  ol {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  li {
    display: grid;
    grid-template-columns: 2.2em 1fr;
    gap: 12px;
    padding: 12px 8px;
    margin-inline: -8px;
    border-top: 1px solid var(--rule);
    scroll-margin: 120px 0 200px;
    transition: background-color 0.6s ease;
  }
  li:first-child {
    border-top: none;
  }
  li.flash {
    background: var(--accent-soft);
    transition: none;
  }
  .num {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--accent-strong);
    padding-top: 0.15em;
    font-variant-numeric: tabular-nums;
  }
  .body {
    min-width: 0;
  }
  .title {
    font-weight: 600;
    line-height: 1.4;
  }
  .meta {
    font-size: 0.84rem;
    color: var(--ink-soft);
  }
  .row {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 14px;
    margin-top: 4px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--ink-faint);
  }
  .row a {
    color: var(--accent-strong);
    text-decoration: none;
    overflow-wrap: anywhere;
  }
  .row a:hover {
    text-decoration: underline;
  }
  .more {
    margin-top: 6px;
    padding: 6px 0;
    border: none;
    background: none;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--accent-strong);
  }
</style>
