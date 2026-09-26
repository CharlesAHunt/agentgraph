<script lang="ts">
  import { tick, type Snippet } from "svelte";
  import { dismiss, type DismissReason } from "./dismiss";

  export interface Option {
    id: string;
    name: string;
    summary: string;
    badge?: string;
  }

  let {
    label,
    legend,
    value,
    display,
    options,
    onselect,
    note,
    disabled = false,
    disabledTitle = "",
    extra,
  }: {
    label: string;
    legend: string;
    value: string;
    display: string;
    options: Option[];
    onselect: (id: string) => void;
    note: string;
    disabled?: boolean;
    disabledTitle?: string;
    /** Rendered below the options, e.g. a text box for a custom choice. */
    extra?: Snippet;
  } = $props();

  let open = $state(false);
  let trigger: HTMLButtonElement;
  let panel = $state<HTMLFieldSetElement>();
  const name = $derived(`picker-${label.toLowerCase()}`);

  async function toggle() {
    open = !open;
    if (!open) return;
    await tick();
    panel?.querySelector<HTMLInputElement>("input:checked")?.focus();
  }

  function dismissed(reason: DismissReason) {
    if (!open) return;
    open = false;
    if (reason === "escape") trigger.focus();
  }
</script>

<div class="picker" use:dismiss={dismissed}>
  <button
    type="button"
    class="chip-trigger"
    bind:this={trigger}
    aria-haspopup="dialog"
    aria-expanded={open}
    {disabled}
    title={disabled ? disabledTitle : undefined}
    onclick={toggle}
  >
    <span class="chip-label">{label}</span>
    <span class="chip-value">{display}</span>
    <svg viewBox="0 0 12 12" aria-hidden="true"><path d="m3 7.5 3-3 3 3" /></svg>
  </button>

  {#if open}
    <fieldset class="popover" bind:this={panel}>
      <legend>{legend}</legend>
      {#each options as o (o.id)}
        <label class="option" class:checked={value === o.id}>
          <input type="radio" {name} value={o.id} checked={value === o.id} onchange={() => onselect(o.id)} />
          <span class="name">{o.name}{#if o.badge}<span class="badge">{o.badge}</span>{/if}</span>
          <span class="summary">{o.summary}</span>
        </label>
      {/each}
      {@render extra?.()}
      <p class="foot">{note}</p>
    </fieldset>
  {/if}
</div>

<style>
  .picker {
    position: relative;
    min-width: 0;
  }
  .popover {
    position: absolute;
    bottom: calc(100% + 8px);
    left: 0;
    z-index: 20;
    width: min(360px, calc(100vw - 2 * var(--gutter)));
    margin: 0;
    padding: 6px 0 0;
    background: var(--surface);
    border: 1px solid var(--rule-strong);
    box-shadow: var(--shadow);
  }
  legend {
    float: left;
    width: 100%;
    padding: 6px 14px 8px;
    font-family: var(--font-mono);
    font-size: 0.64rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-faint);
  }
  .option {
    clear: both;
    display: grid;
    grid-template-columns: auto 1fr;
    column-gap: 10px;
    padding: 7px 14px;
    border-left: 2px solid transparent;
    cursor: pointer;
  }
  .option:hover {
    background: var(--surface-raised);
  }
  .option.checked {
    border-left-color: var(--accent);
    background: var(--surface-raised);
  }
  .option input {
    grid-row: span 2;
    margin: 3px 0 0;
    accent-color: var(--accent);
  }
  .name {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.9rem;
    font-weight: 600;
  }
  .badge {
    padding: 0 5px;
    font-family: var(--font-mono);
    font-size: 0.62rem;
    font-weight: 400;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: var(--surface-sunk);
    color: var(--ink-soft);
  }
  .summary {
    font-size: 0.76rem;
    color: var(--ink-faint);
  }
  .foot {
    clear: both;
    margin: 4px 0 0;
    padding: 7px 14px;
    border-top: 1px solid var(--rule);
    font-family: var(--font-mono);
    font-size: 0.64rem;
    color: var(--ink-faint);
  }
</style>
