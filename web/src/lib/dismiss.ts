export type DismissReason = "pointer" | "escape";

/**
 * Svelte action: call `onDismiss` on a pointer press outside `node` or on Escape.
 * Only an Escape dismissal should move focus back to the trigger.
 */
export function dismiss(node: HTMLElement, onDismiss: (reason: DismissReason) => void) {
  let callback = onDismiss;
  const onPointer = (e: PointerEvent) => {
    if (!node.contains(e.target as Node)) callback("pointer");
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape") callback("escape");
  };
  window.addEventListener("pointerdown", onPointer);
  window.addEventListener("keydown", onKey);
  return {
    update(next: (reason: DismissReason) => void) {
      callback = next;
    },
    destroy() {
      window.removeEventListener("pointerdown", onPointer);
      window.removeEventListener("keydown", onKey);
    },
  };
}
