import DOMPurify from "dompurify";
import katex from "katex";
import { Marked, type TokenizerAndRendererExtension, type Tokens } from "marked";
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

// --- math -------------------------------------------------------------------

// Streaming re-renders the whole answer on every frame; typeset each formula once.
const TEX_CACHE_MAX = 500;
const texCache = new Map<string, string>();

function tex(source: string, displayMode: boolean): string {
  const key = `${displayMode ? "D" : "I"}${source}`;
  let html = texCache.get(key);
  if (html === undefined) {
    // trust:false (the default, stated here on purpose) disables \href, \url and \html* commands.
    html = katex.renderToString(source, { displayMode, throwOnError: false, strict: "ignore", trust: false });
    if (texCache.size >= TEX_CACHE_MAX) texCache.delete(texCache.keys().next().value as string);
    texCache.set(key, html);
  }
  return html;
}

// `$x$` needs non-space just inside both dollars and no digit after the closing one,
// so prose such as "$3 and $15" is left alone.
const DOLLAR_MATH = String.raw`\$(?!\s)((?:\\.|[^\\$\n])+?)(?<!\s)\$(?!\d)`;

const INLINE_MATH: { pattern: RegExp; display: boolean }[] = [
  { pattern: /^\$\$([^$]+?)\$\$/, display: true },
  { pattern: /^\\\[([\s\S]+?)\\\]/, display: true },
  { pattern: /^\\\(([\s\S]+?)\\\)/, display: false },
  { pattern: new RegExp(`^${DOLLAR_MATH}`), display: false },
];

const mathInline: TokenizerAndRendererExtension = {
  name: "math",
  level: "inline",
  start(src) {
    const i = src.search(/\$|\\[([]/);
    return i < 0 ? undefined : i;
  },
  tokenizer(src) {
    for (const { pattern, display } of INLINE_MATH) {
      const m = pattern.exec(src);
      if (m) return { type: "math", raw: m[0], text: m[1].trim(), display };
    }
    return undefined;
  },
  renderer: (token) => tex(token.text, token.display),
};

const mathBlock: TokenizerAndRendererExtension = {
  name: "mathBlock",
  level: "block",
  start(src) {
    const i = src.search(/^ {0,3}(?:\$\$|\\\[)/m);
    return i < 0 ? undefined : i;
  },
  tokenizer(src) {
    const m = /^ {0,3}(?:\$\$([^$]+?)\$\$|\\\[([\s\S]+?)\\\])[ \t]*(?:\n+|$)/.exec(src);
    if (m) return { type: "mathBlock", raw: m[0], text: (m[1] ?? m[2]).trim() };
    return undefined;
  },
  renderer: (token) => `<div class="math-display">${tex(token.text, true)}</div>\n`,
};

/** A kernel's text/latex output (e.g. SymPy's `$\displaystyle …$`) as sanitized display math. */
export function latexOutput(source: string): string {
  const body = source.trim();
  // Strip the delimiters when one pair wraps the whole output.
  const whole = INLINE_MATH.map(({ pattern }) => pattern.exec(body)).find((m) => m?.[0].length === body.length);
  return DOMPurify.sanitize(`<div class="math-display">${tex(whole ? whole[1].trim() : body, true)}</div>`);
}

/** Escaped text with any `$…$` segments typeset, for the stats tiles. */
function textWithMath(text: string): string {
  let out = "";
  let last = 0;
  for (const m of text.matchAll(new RegExp(DOLLAR_MATH, "g"))) {
    out += escape(text.slice(last, m.index)) + tex(m[1], false);
    last = m.index + m[0].length;
  }
  return out + escape(text.slice(last));
}

function statsBlock(text: string): string {
  const tiles = text
    .split("\n")
    .map((line) => line.split("|").map((part) => part.trim()))
    .filter(([value]) => value)
    .slice(0, 4)
    .map(
      ([value, ...label]) =>
        `<div class="stat"><div class="stat-value">${textWithMath(value)}</div>` +
        `<div class="stat-label">${textWithMath(label.join(" | "))}</div></div>`,
    );
  return tiles.length ? `<div class="stats">${tiles.join("")}</div>` : "";
}

const marked = new Marked({
  gfm: true,
  extensions: [mathBlock, mathInline],
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
      node.parentElement?.closest("code, pre, a, button, .stats, .katex")
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
