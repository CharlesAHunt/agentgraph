<script lang="ts">
  import { copier } from "./clipboard.svelte";
  import { latexOutput } from "./markdown";
  import type { Cell } from "./types";

  let { cell, onrun }: { cell: Cell; onrun: (code: string) => Promise<string | null> } = $props();

  let editing = $state(false);
  let draft = $state("");
  let running = $state(false);
  let failure = $state<string | null>(null);
  const clipboard = copier(1200);

  // The server re-encodes images after checking they are PNGs; this is a second, cheap check.
  const isPng = (b64: string) => b64.startsWith("iVBORw0KGgo");

  function edit() {
    draft = cell.code;
    failure = null;
    editing = true;
  }

  async function run() {
    running = true;
    failure = await onrun(draft);
    running = false;
    if (!failure) editing = false;
  }

  function onkeydown(e: KeyboardEvent) {
    if (e.key === "Enter" && (e.shiftKey || e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      void run();
    } else if (e.key === "Escape") {
      editing = false;
    }
  }


  const rows = $derived(Math.min(Math.max(draft.split("\n").length, 2), 18));
</script>

<article class="cell {cell.status}">
  <div class="input">
    <span class="prompt" aria-label="Execution count">In [{cell.execution_count ?? " "}]</span>
    <div class="code-wrap">
      {#if editing}
        <label class="sr-only" for="cell-{cell.id}">Edit code</label>
        <textarea id="cell-{cell.id}" bind:value={draft} {rows} spellcheck="false" {onkeydown}></textarea>
      {:else}
        <pre class="code"><code>{cell.code}</code></pre>
      {/if}
    </div>
    <div class="tools">
      {#if editing}
        <button type="button" class="ghost-button primary" onclick={run} disabled={running || !draft.trim()} title="Shift+Enter">
          {running ? "Running…" : "Run"}
        </button>
        <button type="button" class="ghost-button" onclick={() => (editing = false)} disabled={running}>Cancel</button>
      {:else}
        <button type="button" class="ghost-button" onclick={edit}>Edit</button>
        <button type="button" class="ghost-button" onclick={() => clipboard.copy(cell.code)}>
          {clipboard.copied ? "Copied" : "Copy"}
        </button>
      {/if}
    </div>
  </div>

  {#if failure}<p class="run-failure" role="alert">{failure}</p>{/if}

  {#if cell.outputs.length}
    <div class="outputs">
      {#each cell.outputs as out, i (i)}
        {#if out.type === "stream"}
          <pre class="stream" class:stderr={out.name === "stderr"}>{out.text}</pre>
        {:else if out.type === "text"}
          <pre class="result">{out.text}</pre>
        {:else if out.type === "latex"}
          <div class="latex">{@html latexOutput(out.latex)}</div>
        {:else if out.type === "image" && isPng(out.png)}
          <img src="data:image/png;base64,{out.png}" alt={out.text || "Figure"} />
        {:else if out.type === "error"}
          <pre class="error">{out.ename}: {out.evalue}{out.traceback ? `\n\n${out.traceback}` : ""}</pre>
        {/if}
      {/each}
    </div>
  {/if}

  <footer>
    <span class="status">{cell.status === "ok" ? "ran" : cell.status === "timeout" ? "timed out" : "error"}</span>
    <span>{(cell.duration_ms / 1000).toFixed(cell.duration_ms < 10_000 ? 2 : 0)} s</span>
    {#if cell.edited}<span class="edited">edited by you</span>{/if}
  </footer>
</article>

<style>
  .cell {
    border: 1px solid var(--rule);
    border-left: 3px solid var(--rule-strong);
    background: var(--surface);
  }
  .cell.error,
  .cell.timeout {
    border-left-color: var(--danger);
  }
  .input {
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 10px;
    align-items: start;
    padding: 8px 10px;
    background: var(--surface-raised);
  }
  .prompt {
    padding-top: 3px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--accent-strong);
    white-space: pre;
    font-variant-numeric: tabular-nums;
  }
  .code-wrap {
    min-width: 0;
  }
  .code,
  textarea {
    margin: 0;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    line-height: 1.5;
  }
  .code {
    overflow-x: auto;
    white-space: pre;
  }
  textarea {
    width: 100%;
    resize: vertical;
    padding: 6px 8px;
    border: 1px solid var(--accent);
    background: var(--surface);
  }
  textarea:focus {
    outline: none;
  }
  .tools {
    display: flex;
    gap: 4px;
  }
  .tools button {
    padding: 2px 8px;
    font-size: 0.68rem;
  }
  .tools button.primary,
  .tools button.primary:hover:not(:disabled) {
    background: var(--accent);
    border-color: var(--accent);
    color: var(--on-accent);
  }
  .run-failure {
    margin: 0;
    padding: 6px 12px;
    background: var(--danger-soft);
    color: var(--danger);
    font-size: 0.8rem;
  }
  .outputs {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 10px 12px 10px 58px;
    border-top: 1px solid var(--rule);
  }
  .outputs pre {
    margin: 0;
    overflow-x: auto;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .stream.stderr,
  .error {
    color: var(--danger);
  }
  .latex :global(.katex-display) {
    text-align: left;
  }
  img {
    display: block;
    max-width: 100%;
    height: auto;
    background: #fff;
  }
  footer {
    display: flex;
    gap: 12px;
    padding: 4px 12px 5px 58px;
    border-top: 1px solid var(--rule);
    font-family: var(--font-mono);
    font-size: 0.64rem;
    color: var(--ink-faint);
  }
  .edited {
    color: var(--accent-strong);
  }
  @media (max-width: 560px) {
    .input {
      grid-template-columns: 1fr;
    }
    .outputs,
    footer {
      padding-left: 12px;
    }
  }
</style>
