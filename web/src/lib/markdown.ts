import DOMPurify from "dompurify";
import { Marked, type Tokens } from "marked";
import { matchPaper } from "./sources";
import type { Paper } from "./types";

DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName === "A") {
    node.setAttribute("target", "_blank");
    node.setAttribute("rel", "noopener noreferrer");
  }
});

const escape = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

function isClosed(raw: string): boolean {
  const fence = raw.trimStart().match(/^(`{3,}|~{3,})/)?.[1];
  return !!fence && raw.trimEnd().endsWith(fence) && raw.trim().length > fence.length * 2;
}

function statsBlock(text: string): string {
  const tiles = text
    .split("\n")
    .map((line) => line.split("|").map((part) => part.trim()))
    .filter(([value]) => value)
    .slice(0, 4)
    .map(
      ([value, ...label]) =>
        `<div class="stat"><div class="stat-value">${escape(value)}</div>` +
        `<div class="stat-label">${escape(label.join(" | "))}</div></div>`,
    );
  return tiles.length ? `<div class="stats">${tiles.join("")}</div>` : "";
}

const marked = new Marked({
  gfm: true,
  renderer: {
    code(token: Tokens.Code) {
      const lang = (token.lang ?? "").trim().toLowerCase();
      if (lang === "stats") return statsBlock(token.text);
      if (lang === "mermaid") {
        return (
          `<figure class="diagram" data-complete="${isClosed(token.raw)}">` +
          `<div class="diagram-canvas"></div><pre class="diagram-src">${escape(token.text)}</pre></figure>`
        );
      }
      return false;
    },
  },
});

// Author-year citations such as "Ernst et al., 2024, p. 2" or "Shukla, 2026, pp. 13, 32".
const CITATION =
  /(?<![\p{L}])(\p{Lu}[\p{L}'’-]+)(?:\s+et\s+al\.|\s+(?:and|&)\s+\p{Lu}[\p{L}'’-]+)?,\s+(\d{4})(?:,\s+pp?\.\s*\d+(?:\s*[–-]\s*\d+)?(?:,\s*\d+)*)?/gu;

function linkCitations(root: DocumentFragment, papers: Paper[]): void {
  if (!papers.length) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) =>
      node.parentElement?.closest("code, pre, a, button, .stats")
        ? NodeFilter.FILTER_REJECT
        : NodeFilter.FILTER_ACCEPT,
  });
  const nodes: Text[] = [];
  while (walker.nextNode()) nodes.push(walker.currentNode as Text);

  for (const node of nodes) {
    const text = node.data;
    const out = document.createDocumentFragment();
    let last = 0;
    for (const m of text.matchAll(CITATION)) {
      const paper = matchPaper(papers, m[1], Number(m[2]));
      if (!paper || m.index === undefined) continue;
      out.append(text.slice(last, m.index));
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "cite";
      btn.dataset.paper = String(paper.n);
      btn.title = paper.title;
      btn.textContent = m[0];
      const n = document.createElement("span");
      n.className = "cite-n";
      n.textContent = String(paper.n);
      btn.append(n);
      out.append(btn);
      last = m.index + m[0].length;
    }
    if (last === 0) continue;
    out.append(text.slice(last));
    node.replaceWith(out);
  }
}

/** Markdown from the model -> sanitized HTML with citation chips. */
export function renderMarkdown(text: string, papers: Paper[]): string {
  const html = marked.parse(text, { async: false }) as string;
  const fragment = DOMPurify.sanitize(html, { RETURN_DOM_FRAGMENT: true });
  linkCitations(fragment, papers);
  const host = document.createElement("div");
  host.append(fragment);
  return host.innerHTML;
}
