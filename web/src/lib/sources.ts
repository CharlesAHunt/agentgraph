import type { Paper, Source } from "./types";

/** Group excerpts by paper. Numbering follows first retrieval, so it never shifts mid-stream. */
export function groupPapers(sources: Source[]): Paper[] {
  const byKey = new Map<string, Paper>();
  for (const s of sources) {
    const key = s.arxiv_id || s.doi || s.title;
    let paper = byKey.get(key);
    if (!paper) {
      paper = {
        n: byKey.size + 1,
        key,
        title: s.title,
        authors: s.authors,
        year: s.year,
        doi: s.doi,
        arxiv_id: s.arxiv_id,
        cites: [],
      };
      byKey.set(key, paper);
    }
    if (!paper.cites.some((c) => c.section === s.section && c.page === s.page)) {
      paper.cites.push({ section: s.section, page: s.page });
    }
  }
  return [...byKey.values()];
}

export function surname(author: string): string {
  const parts = author.trim().split(/\s+/);
  return (parts[parts.length - 1] ?? "").replace(/[.,]/g, "").toLowerCase();
}

/** Find the paper an author-year citation refers to. */
export function matchPaper(papers: Paper[], name: string, year: number): Paper | undefined {
  const target = name.toLowerCase();
  return papers.find((p) => p.year === year && p.authors.length > 0 && surname(p.authors[0]) === target);
}

export function paperLink(p: Paper): { href: string; label: string } | null {
  if (p.doi) return { href: `https://doi.org/${encodeURIComponent(p.doi).replace(/%2F/g, "/")}`, label: p.doi };
  if (p.arxiv_id) return { href: `https://arxiv.org/abs/${encodeURIComponent(p.arxiv_id)}`, label: `arXiv:${p.arxiv_id}` };
  return null;
}

export function shortAuthors(authors: string[]): string {
  if (authors.length <= 3) return authors.join(", ");
  return `${authors.slice(0, 3).join(", ")}, et al.`;
}
