<script lang="ts">
  import { tick } from "svelte";
  import Activity from "./Activity.svelte";
  import Answer from "./Answer.svelte";
  import Sources from "./Sources.svelte";
  import { chat } from "./chat.svelte";
  import { groupPapers } from "./sources";
  import type { Turn } from "./types";

  let { turn, last }: { turn: Turn; last: boolean } = $props();

  const papers = $derived(groupPapers(turn.sources));
  let flash = $state<number | null>(null);
  let copied = $state(false);
  let flashTimer: ReturnType<typeof setTimeout>;

  async function cite(n: number) {
    flash = n;
    await tick();
    document.getElementById(`t${turn.id}-p${n}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    clearTimeout(flashTimer);
    flashTimer = setTimeout(() => (flash = null), 1600);
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(turn.text);
      copied = true;
      setTimeout(() => (copied = false), 1500);
    } catch {
      /* clipboard blocked; nothing useful to do */
    }
  }
</script>

<article class="turn">
  <h2 class="question">{turn.question}</h2>

  <Activity {turn} />

  {#if turn.reasoning}
    <details class="reasoning">
      <summary>Reasoning</summary>
      <p>{turn.reasoning}</p>
    </details>
  {/if}

  {#if turn.text}
    <div class="brief" class:live={turn.status === "streaming"}>
      <Answer text={turn.text} {papers} streaming={turn.status === "streaming"} oncite={cite} />
      {#if papers.length}
        <Sources {papers} turnId={turn.id} {flash} />
      {/if}
      {#if turn.status === "done"}
        <div class="actions">
          <button type="button" onclick={copy}>{copied ? "Copied" : "Copy markdown"}</button>
        </div>
      {/if}
    </div>
  {/if}

  {#if turn.status === "error"}
    <div class="notice error" role="alert">
      <p>{turn.error}</p>
      {#if last}<button type="button" onclick={() => chat.retry()}>Try again</button>{/if}
    </div>
  {:else if turn.status === "stopped"}
    <div class="notice">
      <p>Stopped. This question won't be sent as context for the next one.</p>
      {#if last}<button type="button" onclick={() => chat.retry()}>Ask again</button>{/if}
    </div>
  {/if}
</article>

<style>
  .turn {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .question {
    margin: 0;
    font-size: clamp(1.35rem, 3.2vw, 1.75rem);
    font-weight: 500;
    line-height: 1.2;
    letter-spacing: -0.01em;
  }
  .brief {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-top: 3px solid var(--accent);
    box-shadow: var(--shadow);
    padding: clamp(18px, 4vw, 36px);
  }
  .reasoning {
    border-left: 2px solid var(--rule-strong);
    padding-left: 12px;
    color: var(--ink-faint);
  }
  .reasoning summary {
    cursor: pointer;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .reasoning p {
    margin: 8px 0 0;
    max-width: 66ch;
    font-size: 0.86rem;
    white-space: pre-wrap;
  }
  .actions {
    display: flex;
    justify-content: flex-end;
    margin-top: 20px;
  }
  .actions button,
  .notice button {
    padding: 6px 12px;
    border: 1px solid var(--rule-strong);
    background: var(--surface);
    font-family: var(--font-mono);
    font-size: 0.74rem;
    color: var(--ink-soft);
  }
  .actions button:hover,
  .notice button:hover {
    border-color: var(--accent);
    color: var(--accent-strong);
  }
  .notice {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px 16px;
    padding: 12px 16px;
    border: 1px solid var(--rule);
    background: var(--surface-raised);
    font-size: 0.9rem;
    color: var(--ink-soft);
  }
  .notice p {
    margin: 0;
  }
  .notice.error {
    border-color: var(--danger);
    background: var(--danger-soft);
    color: var(--danger);
  }
</style>
