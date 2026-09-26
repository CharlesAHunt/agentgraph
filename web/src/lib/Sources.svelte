<script lang="ts">
  import { paperLink, shortAuthors } from "./sources";
  import type { Paper } from "./types";

  let { papers, turnId, flash }: { papers: Paper[]; turnId: number; flash: number | null } = $props();

  const COLLAPSED = 6;
  let expanded = $state(false);
  const shown = $derived(expanded ? papers : papers.slice(0, COLLAPSED));

  // A citation click on a hidden paper expands the list so it can be scrolled to.
  $effect(() => {
    if (flash !== null && flash > COLLAPSED) expanded = true;
  });
</script>

<section class="sources" aria-label="Sources">
  <h3>Sources <span>{papers.length} {papers.length === 1 ? "paper" : "papers"}</span></h3>
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
</section>

<style>
  .sources {
    margin-top: 28px;
    padding-top: 18px;
    border-top: 1px solid var(--rule);
  }
  h3 {
    margin: 0 0 6px;
    font-size: 1rem;
    font-weight: 600;
  }
  h3 span {
    margin-left: 6px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 400;
    letter-spacing: 0.06em;
    color: var(--ink-faint);
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
