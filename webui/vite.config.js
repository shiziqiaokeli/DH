import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 7860,
    proxy: {
      '/chat': 'http://127.0.0.1:8000',
      '/history': 'http://127.0.0.1:8000',
      '/sessions': 'http://127.0.0.1:8000',
      '/delete': 'http://127.0.0.1:8000',
      '/rag': 'http://127.0.0.1:8000',
      '/knowledgebases': 'http://127.0.0.1:8000',
      '/prompts': 'http://127.0.0.1:8000',
      '/voice_models': 'http://127.0.0.1:8000',
      '/refer_audios': 'http://127.0.0.1:8000',
      '/settings': 'http://127.0.0.1:8000',
      '/tts_proxy': 'http://127.0.0.1:8000',
    },
  },
})