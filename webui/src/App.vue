<script setup>
import { onMounted } from 'vue'
import { useChatStore } from './stores/chat.js'
import Sidebar from './components/Sidebar.vue'
import ChatArea from './components/ChatArea.vue'

const store = useChatStore()

onMounted(async () => {
  store.init()
  await Promise.all([
    store.loadHistory(),
    store.loadSessions(),
    store.loadVoiceMode(),
  ])
})
</script>

<template>
  <div class="app-layout">
    <Sidebar />
    <ChatArea />
  </div>
</template>

<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; overflow: hidden; }
.app-layout { display: flex; height: 100vh; width: 100vw; }
</style>