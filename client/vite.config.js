import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Backend runs on http://localhost:8000 when started with `python kmrl_unified_system.py --api`
      '/api/analyze': 'http://localhost:8000',
      '/classify': 'http://localhost:8000',
      '/embed': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/metrics': 'http://localhost:8000',
    },
  },
})
