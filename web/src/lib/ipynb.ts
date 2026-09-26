import type { Cell, CellOutput, Turn } from "./types";

/** The sandbox runs this before any cell, so an exported notebook starts with it. */
const SETUP = "%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt";

let counter = 0;
const newId = () => `lgraph-${Date.now().toString(36)}-${(counter++).toString(36)}`;

const markdown = (source: string) => ({ cell_type: "markdown", id: newId(), metadata: {}, source });

function output(o: CellOutput, count: number | null): Record<string, unknown> {
  switch (o.type) {
    case "stream":
      return { output_type: "stream", name: o.name, text: o.text };
    case "text":
      return { output_type: "execute_result", execution_count: count, metadata: {}, data: { "text/plain": o.text } };
    case "latex":
      return { output_type: "display_data", metadata: {}, data: { "text/latex": o.latex, "text/plain": o.text } };
    case "image":
      return { output_type: "display_data", metadata: {}, data: { "image/png": o.png, "text/plain": o.text } };
    case "error":
      return { output_type: "error", ename: o.ename, evalue: o.evalue, traceback: o.traceback.split("\n") };
  }
}

const code = (cell: Pick<Cell, "code" | "execution_count" | "outputs"> & { id?: string }) => ({
  cell_type: "code",
  id: cell.id ?? newId(),
  metadata: {},
  execution_count: cell.execution_count,
  source: cell.code,
  outputs: cell.outputs.map((o) => output(o, cell.execution_count)),
});

/** The conversation as an nbformat 4 notebook: each question, its cells, then the answer. */
export function buildNotebook(turns: Turn[]): string {
  const cells: unknown[] = [code({ code: SETUP, execution_count: null, outputs: [] })];
  for (const turn of turns) {
    if (!turn.cells.length && !turn.text) continue;
    cells.push(markdown(`## ${turn.question}`));
    cells.push(...turn.cells.map(code));
    if (turn.text) cells.push(markdown(turn.text));
  }
  return JSON.stringify(
    {
      nbformat: 4,
      nbformat_minor: 5,
      metadata: {
        kernelspec: { name: "python3", display_name: "Python 3", language: "python" },
        language_info: { name: "python" },
      },
      cells,
    },
    null,
    1,
  );
}

export function downloadNotebook(turns: Turn[]): void {
  const blob = new Blob([buildNotebook(turns)], { type: "application/x-ipynb+json" });
  const url = URL.createObjectURL(blob);
  const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "");
  const link = document.createElement("a");
  link.href = url;
  link.download = `lgraph-${stamp}.ipynb`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
