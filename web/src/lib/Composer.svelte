<script lang="ts">
  import { onMount } from "svelte";

  let { busy, onsend, onstop }: { busy: boolean; onsend: (q: string) => void; onstop: () => void } = $props();

  let value = $state("");
  let field: HTMLTextAreaElement;

  onMount(() => field.focus());

  function resize() {
    field.style.height = "auto";
    field.style.height = `${Math.min(field.scrollHeight, 200)}px`;
  }

  function submit() {
    if (!value.trim() || busy) return;
    onsend(value);
    value = "";
    queueMicrotask(resize);
  }

  function onkeydown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      submit();
    }
  }
</script>

<div class="dock">
  <form
    class="composer"
    onsubmit={(e) => {
      e.preventDefault();
      submit();
    }}
  >
    <label class="sr-only" for="question">Ask a question about the corpus</label>
    <textarea
      id="question"
      bind:this={field}
      bind:value
      rows="1"
      placeholder="Ask a research question…"
      {onkeydown}
      oninput={resize}
    ></textarea>
    {#if busy}
      <button type="button" class="stop" onclick={onstop}>
        <span class="square" aria-hidden="true"></span>Stop
      </button>
    {:else}
      <button type="submit" class="send" disabled={!value.trim()}>Ask</button>
    {/if}
  </form>
  <p class="hint">Enter to send · Shift+Enter for a new line</p>
</div>

<style>
  .dock {
    position: fixed;
    inset: auto 0 0 0;
    padding: 14px var(--gutter) calc(10px + env(safe-area-inset-bottom, 0px));
    background: linear-gradient(to bottom, transparent, var(--bg) 32%);
    pointer-events: none;
  }
  .composer,
  .hint {
    max-width: var(--column);
    margin-inline: auto;
    pointer-events: auto;
  }
  .composer {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    padding: 8px 8px 8px 16px;
    background: var(--surface);
    border: 1px solid var(--rule-strong);
    box-shadow: var(--shadow);
    transition: border-color 0.15s ease;
  }
  .composer:focus-within {
    border-color: var(--accent);
  }
  textarea {
    flex: 1;
    min-width: 0;
    resize: none;
    border: none;
    background: none;
    padding: 8px 0;
    line-height: 1.45;
    max-height: 200px;
  }
  textarea:focus {
    outline: none;
  }
  textarea::placeholder {
    color: var(--ink-faint);
  }
  button {
    flex: none;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    height: 38px;
    padding: 0 18px;
    border: 1px solid transparent;
    font-weight: 600;
    font-size: 0.9rem;
  }
  .send {
    background: var(--accent);
    color: var(--on-accent);
  }
  .send:hover:not(:disabled) {
    background: var(--accent-strong);
  }
  .send:disabled {
    background: var(--surface-sunk);
    color: var(--ink-faint);
    cursor: default;
  }
  .stop {
    background: var(--surface);
    border-color: var(--rule-strong);
    color: var(--ink);
  }
  .stop:hover {
    border-color: var(--accent);
  }
  .square {
    width: 9px;
    height: 9px;
    background: var(--accent);
  }
  .hint {
    margin-top: 6px;
    text-align: center;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--ink-faint);
  }
  @media (max-width: 480px) {
    .hint {
      display: none;
    }
  }
</style>
