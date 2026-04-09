import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api } from '../api/index.js'

const LS_KEY = 'dh_chat_session'

export const useChatStore = defineStore('chat', () => {
  const sessionId    = ref(null)
  const sessions     = ref([])          // [{session_id, title}]
  const messages     = ref([])          // [{role, content}]
  const displayNames = ref({})          // sid -> 显示名
  const isStreaming  = ref(false)
  const isVoiceMode  = ref(false)
  const audioUrl     = ref(null)

  // ── 初始化 ───────────────────────────────────────────
  function init() {
    try {
      const saved = JSON.parse(localStorage.getItem(LS_KEY) || '{}')
      sessionId.value = saved.session_id || crypto.randomUUID()
    } catch {
      sessionId.value = crypto.randomUUID()
    }
    _persist()
  }

  function _persist() {
    localStorage.setItem(LS_KEY, JSON.stringify({ session_id: sessionId.value }))
  }

  function getDisplayName(sid) {
    return displayNames.value[sid] || sid
  }

  // ── 会话列表 ─────────────────────────────────────────
  async function loadSessions() {
    try {
      const items = await api.getSessions()
      sessions.value = items
      const merged = {}
      for (const it of items) merged[it.session_id] = it.title || it.session_id
      displayNames.value = { ...displayNames.value, ...merged }
    } catch { sessions.value = [] }
  }

  async function loadHistory(sid) {
    try {
      messages.value = await api.getHistory(sid ?? sessionId.value)
    } catch { messages.value = [] }
  }

  // ── 操作 ─────────────────────────────────────────────
  async function switchSession(sid) {
    sessionId.value = sid
    _persist()
    await loadHistory(sid)
  }

  async function newChat() {
    sessionId.value = crypto.randomUUID()
    _persist()
    messages.value = []
    await loadSessions()
  }

  async function sendMessage(text) {
    if (!text.trim() || isStreaming.value) return
    messages.value.push({ role: 'user', content: text })
    messages.value.push({ role: 'assistant', content: '' })
    isStreaming.value = true
    audioUrl.value = null
    try {
      for await (const chunk of api.streamChat(text, sessionId.value)) {
        messages.value[messages.value.length - 1].content += chunk
      }
    } catch (e) {
      messages.value[messages.value.length - 1].content = `❌ ${e.message}`
    } finally {
      isStreaming.value = false
    }
    // TTS
    if (isVoiceMode.value) {
      const last = messages.value[messages.value.length - 1]
      if (last.role === 'assistant' && last.content.trim()) {
        audioUrl.value = api.getTtsUrl(last.content.trim())
      }
    }
    await loadSessions()
  }

  async function deleteCurrentSession() {
    if (sessionId.value) {
      await api.deleteSession(sessionId.value)
      delete displayNames.value[sessionId.value]
    }
    sessionId.value = crypto.randomUUID()
    _persist()
    messages.value = []
    await loadSessions()
  }

  async function renameSession(newName) {
    if (!sessionId.value || !newName.trim()) return
    await api.updateSessionTitle(sessionId.value, newName.trim())
    displayNames.value[sessionId.value] = newName.trim()
    await loadSessions()
  }

  async function loadVoiceMode() {
    try {
      const d = await api.getVoiceMode()
      isVoiceMode.value = d.is_voice_mode
    } catch { }
  }

  async function toggleVoiceMode() {
    try {
      const d = await api.toggleVoiceMode()
      isVoiceMode.value = d.is_voice_mode
    } catch { }
  }

  return {
    sessionId, sessions, messages, displayNames,
    isStreaming, isVoiceMode, audioUrl,
    init, getDisplayName, loadSessions, loadHistory,
    switchSession, newChat, sendMessage,
    deleteCurrentSession, renameSession,
    loadVoiceMode, toggleVoiceMode,
  }
})