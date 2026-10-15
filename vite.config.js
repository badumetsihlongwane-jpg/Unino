import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  root: '.',
  publicDir: 'public',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: process.env.NODE_ENV !== 'production',
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html')
      }
    }
  },
  server: {
    port: 5173,
    host: true
  }
});
