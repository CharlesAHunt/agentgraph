<script lang="ts">
  import { fetchUsage } from "./api";
  import { dismiss, type DismissReason } from "./dismiss";
  import type { UsageReport } from "./types";

  /** Bump to refetch, e.g. after each finished answer. */
  let { refresh = 0 }: { refresh?: number } = $props();

  let report = $state<UsageReport | null>(null);
  let error = $state<string | null>(null);
  let loading = $state(false);
  let updated = $state<Date | null>(null);
  let open = $state(false);
  let trigger: HTMLButtonElement;

  const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
  function money(n: number): string {
    if (n > 0 && n < 0.01) return "<$0.01";
    return usd.format(n);
  }

  // The badge shows money actually in the account; a key's limit is not a balance.
  const headline = $derived.by(() => {
    if (!report) return null;
    const { key, credits } = report;
    if (credits) {
      return { text: `${money(credits.remaining)} balance`, low: credits.total > 0 && credits.remaining / credits.total < 0.1 };
    }
    return { text: `${money(key.usage)} spent`, low: false };
  });

  const resets: Record<string, string> = { daily: "resets daily", weekly: "resets weekly", monthly: "resets monthly" };

  async function load() {
    loading = true;
    const result = await fetchUsage();
    loading = false;
    if ("report" in result) {
      report = result.report;
      error = null;
      updated = new Date();
    } else {
      error = result.error;
    }
  }

  $effect(() => {
    void refresh;
    void load();
  });

  function toggle() {
    open = !open;
    if (open) void load();
  }

  function dismissed(reason: DismissReason) {
    if (!open) return;
    open = false;
    if (reason === "escape") trigger.focus();
  }

  const pct = (part: number, whole: number) => `${Math.min(100, Math.max(0, (part / whole) * 100))}%`;
</script>

<div class="spend" use:dismiss={dismissed}>
  <button
    type="button"
    class="badge"
    class:low={headline?.low}
    class:muted={!headline}
    bind:this={trigger}
    aria-haspopup="dialog"
    aria-expanded={open}
    title={report?.credits_note ?? undefined}
    onclick={toggle}
  >
    {#if headline}{headline.text}{:else if error}spend unavailable{:else}loading spend…{/if}
  </button>

  {#if open}
    <div class="panel" role="dialog" aria-label="OpenRouter spend">
      {#if report}
        {@const { key, credits } = report}
        <section>
          <h4>Account balance</h4>
          {#if credits}
            <p class="big" class:low={headline?.low}>{money(credits.remaining)} <span>available</span></p>
            {#if credits.total > 0}
              <div class="meter" class:low={headline?.low} aria-hidden="true"><div style:width={pct(credits.used, credits.total)}></div></div>
            {/if}
            <dl>
              <div><dt>Used</dt><dd>{money(credits.used)}</dd></div>
              <div><dt>Purchased</dt><dd>{money(credits.total)}</dd></div>
            </dl>
          {:else}
            <p class="note">{report.credits_note}</p>
          {/if}
        </section>

        <section>
          <h4>This API key {#if key.is_free_tier}<span class="chip">free tier</span>{/if}</h4>
          <dl>
            <div><dt>Spent in total</dt><dd>{money(key.usage)}</dd></div>
            <div><dt>Today</dt><dd>{money(key.usage_daily)}</dd></div>
            <div><dt>This week</dt><dd>{money(key.usage_weekly)}</dd></div>
            <div><dt>This month</dt><dd>{money(key.usage_monthly)}</dd></div>
          </dl>
          {#if key.limit !== null}
            <div class="limit">
              <div class="row">
                <span>Limit {money(key.limit)} · {resets[key.limit_reset ?? ""] ?? "doesn't reset"}</span>
                {#if key.limit_remaining !== null}<span>{money(key.limit_remaining)} left</span>{/if}
              </div>
              {#if key.limit > 0 && key.limit_remaining !== null}
                <div class="meter" aria-hidden="true"><div style:width={pct(key.limit - key.limit_remaining, key.limit)}></div></div>
              {/if}
            </div>
          {:else}
            <p class="quiet">No spending limit is set on this key.</p>
          {/if}
        </section>
      {/if}

      {#if error}<p class="note error">{error}</p>{/if}

      <footer>
        <span>{updated ? `Updated ${updated.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` : ""}</span>
        <button type="button" onclick={load} disabled={loading}>{loading ? "Refreshing…" : "Refresh"}</button>
      </footer>
    </div>
  {/if}
</div>

<style>
  .spend {
    position: relative;
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
  .badge:hover,
  .badge[aria-expanded="true"] {
    border-color: var(--accent);
  }
  .badge.low {
    border-color: var(--danger);
    color: var(--danger);
  }
  .badge.muted {
    color: var(--ink-faint);
  }
  .panel {
    position: absolute;
    top: calc(100% + 8px);
    right: 0;
    z-index: 30;
    width: min(330px, calc(100vw - 2 * var(--gutter)));
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 16px;
    background: var(--surface);
    border: 1px solid var(--rule-strong);
    box-shadow: var(--shadow);
  }
  section {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  section + section {
    padding-top: 14px;
    border-top: 1px solid var(--rule);
  }
  h4 {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0;
    font-family: var(--font-mono);
    font-size: 0.66rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-faint);
  }
  .chip {
    padding: 0 5px;
    background: var(--accent-soft);
    color: var(--accent-strong);
    letter-spacing: 0.06em;
  }
  .big {
    margin: 0;
    font-family: var(--font-display);
    font-size: 1.7rem;
    font-weight: 600;
    line-height: 1.1;
    color: var(--accent-strong);
    font-variant-numeric: tabular-nums;
  }
  .big.low {
    color: var(--danger);
  }
  .big span {
    font-family: var(--font-body);
    font-size: 0.8rem;
    font-weight: 400;
    color: var(--ink-soft);
  }
  .meter {
    height: 6px;
    background: var(--surface-sunk);
    overflow: hidden;
  }
  .meter div {
    height: 100%;
    background: var(--accent);
  }
  .meter.low div {
    background: var(--danger);
  }
  dl {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px 16px;
    margin: 0;
  }
  dl div {
    display: flex;
    flex-direction: column;
  }
  dt {
    font-size: 0.72rem;
    color: var(--ink-faint);
  }
  dd {
    margin: 0;
    font-family: var(--font-mono);
    font-size: 0.85rem;
    font-variant-numeric: tabular-nums;
  }
  .limit {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 4px;
  }
  .limit .row {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    font-size: 0.78rem;
    color: var(--ink-soft);
    font-variant-numeric: tabular-nums;
  }
  .quiet {
    margin: 4px 0 0;
    font-size: 0.78rem;
    color: var(--ink-faint);
  }
  .note {
    margin: 0;
    padding: 8px 10px;
    background: var(--surface-raised);
    border-left: 2px solid var(--rule-strong);
    font-size: 0.78rem;
    color: var(--ink-soft);
  }
  .note.error {
    border-left-color: var(--danger);
    color: var(--danger);
  }
  footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--ink-faint);
  }
  footer button {
    padding: 4px 10px;
    border: 1px solid var(--rule-strong);
    background: var(--surface);
    font: inherit;
    color: var(--ink-soft);
  }
  footer button:hover:not(:disabled) {
    border-color: var(--accent);
    color: var(--accent-strong);
  }
</style>
