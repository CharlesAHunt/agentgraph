/** Copy text to the clipboard and remember briefly that it happened, for a "Copied" label. */
export function copier(ms = 1500) {
  let copied = $state(false);
  return {
    get copied() {
      return copied;
    },
    async copy(text: string) {
      try {
        await navigator.clipboard.writeText(text);
        copied = true;
        setTimeout(() => (copied = false), ms);
      } catch {
        /* clipboard blocked; nothing useful to do */
      }
    },
  };
}
