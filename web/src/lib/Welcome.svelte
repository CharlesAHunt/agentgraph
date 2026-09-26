<script lang="ts">
  import type { CorpusInfo } from "./types";

  let { corpus, onpick }: { corpus: CorpusInfo | null; onpick: (q: string) => void } = $props();

  const examples = [
    { topic: "Power exhaust", q: "What limits divertor heat flux in tokamaks?" },
    { topic: "Scaling laws", q: "How does the scrape-off-layer heat-flux width λq scale with poloidal field?" },
    { topic: "Detachment", q: "How is divertor detachment controlled in real time without degrading the core?" },
    { topic: "Configurations", q: "How does negative triangularity change the power exhaust problem?" },
  ];
</script>

<section class="welcome">
  <p class="eyebrow">Research assistant · arXiv physics.plasm-ph</p>
  <h1>Ask the fusion literature.</h1>
  <p class="lede">
    Answers are written from excerpts retrieved out of
    {#if corpus?.enabled}
      <strong>{corpus.papers.toLocaleString()} papers</strong>
    {:else}
      the ingested corpus
    {/if}
    on tokamaks, stellarators and confinement, with every claim cited to a page you can open.
  </p>
  {#if corpus && !corpus.enabled}
    <p class="warn">Retrieval is off: no corpus was found at <code>data/lancedb</code>, so answers won't cite papers.</p>
  {/if}
  <ul class="examples">
    {#each examples as e (e.q)}
      <li>
        <button type="button" onclick={() => onpick(e.q)}>
          <span class="topic">{e.topic}</span>
          <span class="q">{e.q}</span>
        </button>
      </li>
    {/each}
  </ul>
</section>

<style>
  .welcome {
    padding-top: clamp(24px, 8vh, 80px);
  }
  .eyebrow {
    margin: 0 0 12px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent-strong);
  }
  h1 {
    margin: 0;
    font-size: clamp(2rem, 6vw, 3.1rem);
    font-weight: 500;
    line-height: 1.05;
    letter-spacing: -0.02em;
  }
  .lede {
    max-width: 56ch;
    margin: 16px 0 0;
    font-size: 1.02rem;
    color: var(--ink-soft);
  }
  .lede strong {
    color: var(--ink);
    font-weight: 600;
  }
  .warn {
    margin: 14px 0 0;
    padding: 10px 14px;
    border-left: 3px solid var(--danger);
    background: var(--danger-soft);
    font-size: 0.88rem;
  }
  .warn code {
    font-family: var(--font-mono);
    font-size: 0.85em;
  }
  .examples {
    list-style: none;
    margin: 32px 0 0;
    padding: 0;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(min(100%, 300px), 1fr));
    gap: 10px;
  }
  .examples button {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 14px 16px;
    border: 1px solid var(--rule);
    background: var(--surface);
    text-align: left;
    transition: border-color 0.15s ease, transform 0.15s ease;
  }
  .examples button:hover {
    border-color: var(--accent);
    transform: translateY(-1px);
  }
  .topic {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-faint);
  }
  .q {
    font-size: 0.95rem;
    line-height: 1.4;
  }
</style>
