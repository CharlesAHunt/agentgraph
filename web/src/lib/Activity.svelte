<script lang="ts">
  import { chat } from "./chat.svelte";
  import { effortName } from "./efforts";
  import { plural } from "./format";
  import { PYTHON_TOOL, type Search, type Turn } from "./types";

  let { turn }: { turn: Turn } = $props();

  const isPython = (tool: string) => tool === PYTHON_TOOL;

  function statusText(s: Search): string {
    if (s.status === "running") return isPython(s.tool) ? "running" : "searching";
    if (s.status === "error") return "failed";
    return isPython(s.tool) ? "ran" : String(s.count);
  }
  const searches = $derived(turn.searches.filter((s) => !isPython(s.tool)));
  const runs = $derived(turn.searches.length - searches.length);
  const excerpts = $derived(searches.reduce((sum, s) => sum + s.count, 0));
  const running = $derived(turn.searches.find((s) => s.status === "running"));
  const label = $derived.by(() => {
    if (turn.status === "streaming") {
      if (running) return isPython(running.tool) ? "Running Python" : "Searching the corpus";
      if (turn.text) return "Writing";
      if (turn.searches.length) return "Reading the results";
      return turn.reasoning ? "Thinking" : "Reading the question";
    }
    const parts: string[] = [];
    if (searches.length) parts.push(`${plural(searches.length, "search", "searches")} · ${excerpts} excerpts`);
    if (runs) parts.push(plural(runs, "Python run"));
    return parts.join(" · ") || "Answered without searching the corpus";
  });
</script>

<div class="activity" aria-live="polite">
  <div class="phase" class:live={turn.status === "streaming"}>
    {#if turn.status === "streaming"}<span class="pulse" aria-hidden="true"></span>{/if}
    {label}
    {#if turn.model}<span class="model" title={turn.model}>· {chat.modelName(turn.model)}</span>{/if}
    {#if turn.role.instructions}<span class="model" title={turn.role.instructions}>· {turn.role.name}</span>{/if}
    {#if turn.effort && turn.effort !== chat.defaultEffort}<span class="model">· {effortName(turn.effort)} effort</span>{/if}
  </div>
  {#if turn.searches.length}
    <ul class="searches">
      {#each turn.searches as s (s.id)}
        <li class="search {s.status}">
          {#if isPython(s.tool)}
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m3 5 3 3-3 3M8 12h5" /></svg>
          {:else}
            <svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="7" cy="7" r="4.5" /><path d="m10.5 10.5 3.5 3.5" /></svg>
          {/if}
          <span class="q">{s.query}</span>
          <span class="n">{statusText(s)}</span>
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
  .model {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--ink-faint);
    text-transform: none;
    letter-spacing: 0.02em;
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
