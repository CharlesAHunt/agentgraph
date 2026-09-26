<script lang="ts">
  import { tick } from "svelte";
  import { chat } from "./chat.svelte";
  import type { ModelInfo } from "./types";

  let open = $state(false);
  let query = $state("");
  let active = $state(0);
  let root: HTMLDivElement;
  let trigger: HTMLButtonElement;
  let search = $state<HTMLInputElement>();
  let list = $state<HTMLUListElement>();

  const current = $derived(chat.models.find((m) => m.id === chat.model));
  const filtered = $derived.by(() => {
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) return chat.models;
    return chat.models.filter((m) => {
      const haystack = `${m.name} ${m.id}`.toLowerCase();
      return terms.every((t) => haystack.includes(t));
    });
  });

  /** OpenRouter names read "Provider: Model". */
  function split(name: string) {
    const i = name.indexOf(": ");
    return i > 0 ? { provider: name.slice(0, i), title: name.slice(i + 2) } : { provider: "", title: name };
  }

  const usd = (n: number) => `$${n >= 100 ? n.toFixed(0) : n.toFixed(2)}`;

  function price(m: ModelInfo): string {
    if (m.prompt_price === null || m.completion_price === null) return "variable price";
    if (m.prompt_price === 0 && m.completion_price === 0) return "free";
    return `${usd(m.prompt_price)} / ${usd(m.completion_price)}`;
  }

  function context(n: number): string {
    return n >= 1_000_000 ? `${+(n / 1_000_000).toFixed(1)}M` : `${Math.round(n / 1000)}K`;
  }

  function reveal() {
    void tick().then(() => list?.querySelector(`[data-i="${active}"]`)?.scrollIntoView({ block: "nearest" }));
  }

  async function toggle() {
    if (open) return close();
    query = "";
    active = Math.max(0, chat.models.findIndex((m) => m.id === chat.model));
    open = true;
    await tick();
    search?.focus();
    reveal();
  }

  function close() {
    open = false;
    trigger.focus();
  }

  function choose(m: ModelInfo | undefined) {
    if (!m) return;
    chat.selectModel(m.id);
    close();
  }

  function onkeydown(e: KeyboardEvent) {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const step = e.key === "ArrowDown" ? 1 : -1;
      active = Math.min(Math.max(active + step, 0), filtered.length - 1);
      reveal();
    } else if (e.key === "Enter") {
      e.preventDefault();
      choose(filtered[active]);
    } else if (e.key === "Escape") {
      e.preventDefault();
      close();
    }
  }

  function onpointerdown(e: PointerEvent) {
    if (open && !root.contains(e.target as Node)) open = false;
  }
</script>

<svelte:window {onpointerdown} />

<div class="picker" bind:this={root}>
  <button
    type="button"
    class="trigger"
    bind:this={trigger}
    aria-haspopup="listbox"
    aria-expanded={open}
    title={chat.model}
    onclick={toggle}
  >
    <span class="label">Model</span>
    <span class="value">{current ? split(current.name).title : chat.model}</span>
    <svg viewBox="0 0 12 12" aria-hidden="true"><path d="m3 7.5 3-3 3 3" /></svg>
  </button>

  {#if open}
    <div class="popover">
      <input
        id="model-search"
        bind:this={search}
        bind:value={query}
        oninput={() => (active = 0)}
        {onkeydown}
        role="combobox"
        aria-label="Search models"
        aria-controls="model-list"
        aria-expanded="true"
        aria-activedescendant={filtered[active] ? `model-opt-${active}` : undefined}
        placeholder="Search {chat.models.length} models by name or provider"
        autocomplete="off"
        spellcheck="false"
      />
      <!-- Keyboard selection happens on the combobox input via aria-activedescendant. -->
      <!-- svelte-ignore a11y_click_events_have_key_events -->
      <ul id="model-list" role="listbox" aria-label="Models" bind:this={list}>
        {#each filtered as m, i (m.id)}
          {@const n = split(m.name)}
          <li
            id="model-opt-{i}"
            role="option"
            aria-selected={m.id === chat.model}
            data-i={i}
            class:active={i === active}
            onclick={() => choose(m)}
            onpointermove={() => (active = i)}
          >
            <div class="top">
              <span class="title">{n.title}</span>
              {#if m.id === chat.defaultModel}<span class="chip">default</span>{/if}
              {#if m.id === chat.model}
                <svg class="check" viewBox="0 0 12 12" aria-hidden="true"><path d="m2.5 6.5 2.5 2.5 4.5-5" /></svg>
              {/if}
            </div>
            <div class="sub">
              <span class="provider">{n.provider || m.id}</span>
              <span class="meta">{price(m)}{m.context_length ? ` · ${context(m.context_length)}` : ""}</span>
            </div>
          </li>
        {:else}
          <li class="empty">No models match “{query}”.</li>
        {/each}
      </ul>
      <p class="foot">Tool-calling models on OpenRouter · $ per 1M tokens in / out · context</p>
    </div>
  {/if}
</div>

<style>
  .picker {
    position: relative;
    min-width: 0;
  }
  .trigger {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    max-width: 100%;
    padding: 4px 8px;
    border: 1px solid var(--rule);
    background: var(--surface);
    font-size: 0.8rem;
  }
  .trigger:hover,
  .trigger[aria-expanded="true"] {
    border-color: var(--accent);
  }
  .label {
    font-family: var(--font-mono);
    font-size: 0.64rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-faint);
  }
  .value {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-weight: 600;
  }
  .trigger svg {
    flex: none;
    width: 10px;
    height: 10px;
    fill: none;
    stroke: var(--ink-faint);
    stroke-width: 1.6;
  }
  .popover {
    position: absolute;
    bottom: calc(100% + 8px);
    left: 0;
    z-index: 20;
    width: min(460px, calc(100vw - 2 * var(--gutter)));
    display: flex;
    flex-direction: column;
    background: var(--surface);
    border: 1px solid var(--rule-strong);
    box-shadow: var(--shadow);
  }
  input {
    margin: 8px;
    padding: 8px 10px;
    border: 1px solid var(--rule);
    background: var(--surface-raised);
    color: var(--ink);
    font: inherit;
    font-size: 0.88rem;
  }
  input:focus {
    outline: none;
    border-color: var(--accent);
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 0 0 4px;
    max-height: min(52vh, 420px);
    overflow-y: auto;
  }
  li[role="option"] {
    padding: 8px 14px;
    cursor: pointer;
    border-left: 2px solid transparent;
  }
  li.active {
    background: var(--surface-raised);
    border-left-color: var(--accent);
  }
  .top {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .title {
    font-size: 0.9rem;
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .chip {
    flex: none;
    padding: 0 5px;
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: var(--surface-sunk);
    color: var(--ink-soft);
  }
  .check {
    flex: none;
    width: 13px;
    height: 13px;
    margin-left: auto;
    fill: none;
    stroke: var(--accent);
    stroke-width: 1.8;
  }
  .sub {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin-top: 1px;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--ink-faint);
  }
  .provider {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .meta {
    flex: none;
    font-variant-numeric: tabular-nums;
  }
  .empty {
    padding: 14px;
    font-size: 0.85rem;
    color: var(--ink-faint);
  }
  .foot {
    margin: 0;
    padding: 7px 14px;
    border-top: 1px solid var(--rule);
    font-family: var(--font-mono);
    font-size: 0.64rem;
    color: var(--ink-faint);
  }
</style>
