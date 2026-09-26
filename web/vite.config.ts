import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

const api = process.env.LGRAPH_API ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      "/chat": api,
      "/papers": api,
      "/health": api,
      "/results": api,
    },
  },
});
