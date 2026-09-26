<script lang="ts">
  import { onMount, tick } from "svelte";
  import Composer from "./lib/Composer.svelte";
  import Spend from "./lib/Spend.svelte";
  import { downloadNotebook } from "./lib/ipynb";
  import Turn from "./lib/Turn.svelte";
  import Welcome from "./lib/Welcome.svelte";
  import { fetchCorpus, fetchModels } from "./lib/api";
  import { chat } from "./lib/chat.svelte";
  import type { CorpusInfo } from "./lib/types";

  let corpus = $state<CorpusInfo | null>(null);
  let stickToBottom = true;
  const finished = $derived(chat.turns.filter((t) => t.status === "done").length);
  const hasCells = $derived(chat.turns.some((t) => t.cells.length > 0));

  onMount(async () => {
    const [info, models] = await Promise.all([fetchCorpus(), fetchModels()]);
    corpus = info;
    if (models) chat.setModels(models);
  });

  function onscroll() {
    const { scrollHeight } = document.documentElement;
    stickToBottom = window.innerHeight + window.scrollY >= scrollHeight - 160;
  }

  // Follow the stream while the reader is at the bottom; stop if they scroll up to read.
  $effect(() => {
    const last = chat.turns.at(-1);
    void [chat.turns.length, last?.text.length, last?.searches.length, last?.status, last?.sources.length];
    if (!stickToBottom) return;
    tick().then(() => window.scrollTo({ top: document.documentElement.scrollHeight }));
  });

  function ask(q: string) {
    stickToBottom = true;
    chat.ask(q);
  }
</script>

<svelte:window {onscroll} />

<header class="bar">
  <div class="inner">
    <div class="brand">
      <svg viewBox="0 0 32 32" aria-hidden="true">
        <ellipse cx="16" cy="16" rx="13" ry="7.5" />
        <circle cx="16" cy="16" r="2.6" />
      </svg>
      <span class="name">lgraph</span>
      <span class="tag">fusion corpus</span>
    </div>
    <div class="right">
      {#if corpus}
        <span class="badge" class:off={!corpus.enabled}>
          {corpus.enabled ? `${corpus.papers.toLocaleString()} papers` : "retrieval off"}
        </span>
      {/if}
      <Spend refresh={finished} />
      {#if chat.turns.length}
        {#if hasCells}
          <button type="button" class="new" onclick={() => downloadNotebook(chat.turns)} title="Download this conversation as a Jupyter notebook">Export .ipynb</button>
        {/if}
        <button type="button" class="new" onclick={() => chat.reset()}>New chat</button>
      {/if}
    </div>
  </div>
</header>

<main>
  {#if chat.turns.length === 0}
    <Welcome {corpus} onpick={ask} />
  {:else}
    {#each chat.turns as turn, i (turn.id)}
      <Turn {turn} last={i === chat.turns.length - 1} />
    {/each}
  {/if}
</main>

<Composer busy={chat.busy} onsend={ask} onstop={() => chat.stop()} />

<style>
  .bar {
    position: sticky;
    top: 0;
    z-index: 10;
    padding-top: env(safe-area-inset-top, 0px);
    background: color-mix(in srgb, var(--bg) 88%, transparent);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--rule);
  }
  .inner {
    max-width: calc(var(--column) + 2 * var(--gutter));
    margin-inline: auto;
    padding: 10px var(--gutter);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 9px;
    min-width: 0;
  }
  .brand svg {
    width: 26px;
    height: 26px;
    flex: none;
  }
  .brand ellipse {
    fill: none;
    stroke: var(--accent);
    stroke-width: 2.4;
  }
  .brand circle {
    fill: var(--accent);
  }
  .name {
    font-family: var(--font-display);
    font-size: 1.2rem;
    font-weight: 600;
    letter-spacing: -0.01em;
  }
  .tag {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-faint);
    white-space: nowrap;
  }
  .right {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .badge {
    padding: 3px 8px;
    border: 1px solid var(--rule);
    background: var(--surface);
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--ink-soft);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .badge.off {
    border-color: var(--danger);
    color: var(--danger);
  }
  .new {
    padding: 5px 12px;
    border: 1px solid var(--rule-strong);
    background: var(--surface);
    font-size: 0.82rem;
    font-weight: 600;
    white-space: nowrap;
  }
  .new:hover {
    border-color: var(--accent);
    color: var(--accent-strong);
  }
  main {
    max-width: calc(var(--column) + 2 * var(--gutter));
    margin-inline: auto;
    padding: 32px var(--gutter) 200px;
    display: flex;
    flex-direction: column;
    gap: 56px;
  }
  @media (max-width: 480px) {
    .tag,
    .badge {
      display: none;
    }
  }
</style>
