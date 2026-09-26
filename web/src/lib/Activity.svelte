<script lang="ts">
  import type { Turn } from "./types";

  let { turn }: { turn: Turn } = $props();

  const excerpts = $derived(turn.searches.reduce((sum, s) => sum + s.count, 0));
  const running = $derived(turn.searches.some((s) => s.status === "running"));
  const label = $derived.by(() => {
    if (turn.status === "streaming") {
      if (running) return "Searching the corpus";
      if (turn.text) return "Writing";
      if (turn.searches.length) return "Reading the excerpts";
      return turn.reasoning ? "Thinking" : "Reading the question";
    }
    if (!turn.searches.length) return "Answered without searching the corpus";
    const n = turn.searches.length;
    return `${n} ${n === 1 ? "search" : "searches"} · ${excerpts} excerpts`;
  });
</script>

<div class="activity" aria-live="polite">
  <div class="phase" class:live={turn.status === "streaming"}>
    {#if turn.status === "streaming"}<span class="pulse" aria-hidden="true"></span>{/if}
    {label}
  </div>
  {#if turn.searches.length}
    <ul class="searches">
      {#each turn.searches as s (s.id)}
        <li class="search {s.status}">
          <svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="7" cy="7" r="4.5" /><path d="m10.5 10.5 3.5 3.5" /></svg>
          <span class="q">{s.query}</span>
          <span class="n">
            {#if s.status === "running"}searching{:else if s.status === "error"}failed{:else}{s.count}{/if}
          </span>
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .activity {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .phase {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-faint);
  }
  .phase.live {
    color: var(--accent-strong);
  }
  .pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1.2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%,
    100% {
      transform: scale(0.7);
      opacity: 0.5;
    }
    50% {
      transform: scale(1);
      opacity: 1;
    }
  }
  .searches {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .search {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    max-width: 100%;
    padding: 5px 6px 5px 9px;
    border: 1px solid var(--rule);
    background: var(--surface);
    font-family: var(--font-mono);
    font-size: 0.76rem;
    color: var(--ink-soft);
    overflow: hidden;
  }
  .search svg {
    flex: none;
    width: 12px;
    height: 12px;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.6;
    stroke-linecap: round;
  }
  .q {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .n {
    flex: none;
    padding: 1px 6px;
    border-radius: 2px;
    background: var(--surface-sunk);
    color: var(--ink-faint);
    font-size: 0.68rem;
    font-variant-numeric: tabular-nums;
  }
  .search.done .n {
    background: var(--accent-soft);
    color: var(--accent-strong);
  }
  .search.error {
    border-color: var(--danger);
    color: var(--danger);
  }
  .search.running::after {
    content: "";
    position: absolute;
    left: 0;
    bottom: 0;
    height: 2px;
    width: 40%;
    background: var(--accent);
    animation: scan 1.1s ease-in-out infinite alternate;
  }
  @keyframes scan {
    from {
      transform: translateX(-10%);
    }
    to {
      transform: translateX(160%);
    }
  }
</style>
