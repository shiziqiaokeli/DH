<script setup>
import { ref } from 'vue'
import { useChatStore } from '../stores/chat.js'
import { api } from '../api/index.js'

const store = useChatStore()
const inputText  = ref('')
const activePanel = ref(null)   // 'kb' | 'prompt' | 't' | 'model' | 'audio' | null

// ── 各面板数据 ────────────────────────────────────────
const kbList        = ref([])
const selectedKbId  = ref(null)
const kbStatus      = ref('')

const promptList        = ref([])
const selectedPromptId  = ref(null)
const promptStatus      = ref('')

const tValue   = ref(0.1)
const tStatus  = ref('')

const modelList        = ref([])
const selectedModelId  = ref(null)
const modelStatus      = ref('')

const audioList        = ref([])
const selectedAudioId  = ref(null)
const audioStatus      = ref('')

// ── 面板开关 ──────────────────────────────────────────
async function togglePanel(name) {
  if (activePanel.value === name) { activePanel.value = null; return }
  activePanel.value = name
  // 打开时拉取最新数据
  try {
    if (name === 'kb') {
      const d = await api.getKnowledgeBases()
      kbList.value       = d.knowledgebases
      selectedKbId.value = d.active_kb_id ? String(d.active_kb_id) : null
    } else if (name === 'prompt') {
      const d = await api.getPrompts()
      promptList.value       = d.prompts
      selectedPromptId.value = d.active_prompt_id ? String(d.active_prompt_id) : null
    } else if (name === 't') {
      const d = await api.getTemperature()
      tValue.value = d.t_value
    } else if (name === 'model') {
      const d = await api.getVoiceModels()
      modelList.value       = d.voice_models
      selectedModelId.value = d.active_model_id ? String(d.active_model_id) : null
    } else if (name === 'audio') {
      const d = await api.getReferAudios()
      audioList.value       = d.refer_audios
      selectedAudioId.value = d.active_audio_id ? String(d.active_audio_id) : null
    }
  } catch { /* 后端未连接时静默处理 */ }
}

// ── 选中操作 ──────────────────────────────────────────
async function selectKb(id) {
  selectedKbId.value = String(id)
  await api.setActiveKb(id)
  kbStatus.value = '✅ 知识库已切换'
}
async function selectPrompt(id) {
  selectedPromptId.value = String(id)
  await api.setActivePrompt(id)
  promptStatus.value = '✅ 提示词已切换'
}
async function saveT() {
  if (tValue.value <= 0) { tStatus.value = '⚠️ 温度必须大于 0'; return }
  await api.setTemperature(tValue.value)
  tStatus.value = `✅ 温度已设为 ${tValue.value}`
}
async function selectModel(id) {
  selectedModelId.value = String(id)
  await api.setActiveModel(id)
  modelStatus.value = '✅ GSV 模型已切换'
}
async function selectAudio(id) {
  selectedAudioId.value = String(id)
  await api.setActiveAudio(id)
  audioStatus.value = '✅ 参考音频已切换'
}

// ── 发送 ──────────────────────────────────────────────
function handleSend() {
  const text = inputText.value.trim()
  if (!text) return
  inputText.value = ''
  store.sendMessage(text)
}
function handleKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
}

const voiceLabel = () => store.isVoiceMode ? '语音输出' : '文本输出'
</script>

<template>
  <div class="input-wrapper" @click.stop>
    <div class="input-card">
      <!-- 输入框 -->
      <textarea
        v-model="inputText"
        class="msg-input"
        placeholder="你想聊些什么"
        rows="1"
        @keydown="handleKeydown"
        @input="e => { e.target.style.height='auto'; e.target.style.height=e.target.scrollHeight+'px' }"
      />

      <!-- 工具栏 -->
      <div class="toolbar">

        <!-- 知识库 -->
        <div class="anchor">
          <button class="mini-btn" @click="togglePanel('kb')">知识库</button>
          <div v-if="activePanel === 'kb'" class="popup-panel">
            <p class="panel-label">选择知识库</p>
            <div v-for="kb in kbList" :key="kb.id"
                 class="radio-item" :class="{active: selectedKbId === String(kb.id)}"
                 @click="selectKb(kb.id)">{{ kb.name }}</div>
            <p v-if="kbStatus" class="pstatus">{{ kbStatus }}</p>
            <button class="btn-sm" @click="activePanel = null">关闭</button>
          </div>
        </div>

        <!-- 提示词 -->
        <div class="anchor">
          <button class="mini-btn" @click="togglePanel('prompt')">提示词</button>
          <div v-if="activePanel === 'prompt'" class="popup-panel">
            <p class="panel-label">选择提示词</p>
            <div v-for="p in promptList" :key="p.id"
                 class="radio-item" :class="{active: selectedPromptId === String(p.id)}"
                 @click="selectPrompt(p.id)">{{ p.name }}</div>
            <p v-if="promptStatus" class="pstatus">{{ promptStatus }}</p>
            <button class="btn-sm" @click="activePanel = null">关闭</button>
          </div>
        </div>

        <!-- 温度 -->
        <div class="anchor">
          <button class="mini-btn" @click="togglePanel('t')">T</button>
          <div v-if="activePanel === 't'" class="popup-panel" style="width:200px">
            <label class="panel-label">温度参数 (&gt;0)</label>
            <input v-model.number="tValue" type="number" step="0.1" min="0.01" class="num-input" />
            <p v-if="tStatus" class="pstatus">{{ tStatus }}</p>
            <div style="display:flex;gap:8px;margin-top:8px">
              <button class="btn-sm primary" @click="saveT">保存</button>
              <button class="btn-sm" @click="activePanel = null">关闭</button>
            </div>
          </div>
        </div>

        <!-- GSV 模型 -->
        <div class="anchor">
          <button class="mini-btn" @click="togglePanel('model')">GSV模型</button>
          <div v-if="activePanel === 'model'" class="popup-panel">
            <p class="panel-label">选择 GSV 模型</p>
            <div v-for="m in modelList" :key="m.id"
                 class="radio-item" :class="{active: selectedModelId === String(m.id)}"
                 @click="selectModel(m.id)">{{ m.name }}</div>
            <p v-if="modelStatus" class="pstatus">{{ modelStatus }}</p>
            <button class="btn-sm" @click="activePanel = null">关闭</button>
          </div>
        </div>

        <!-- 参考音频 -->
        <div class="anchor">
          <button class="mini-btn" @click="togglePanel('audio')">参考音频</button>
          <div v-if="activePanel === 'audio'" class="popup-panel">
            <p class="panel-label">选择参考音频</p>
            <div v-for="a in audioList" :key="a.id"
                 class="radio-item" :class="{active: selectedAudioId === String(a.id)}"
                 @click="selectAudio(a.id)">{{ a.name }}</div>
            <p v-if="audioStatus" class="pstatus">{{ audioStatus }}</p>
            <button class="btn-sm" @click="activePanel = null">关闭</button>
          </div>
        </div>

        <div style="flex:1" />

        <!-- 语音/文本切换 -->
        <button class="mini-btn"
                :class="{ active: store.isVoiceMode }"
                @click="store.toggleVoiceMode">{{ voiceLabel() }}</button>

        <!-- 发送 -->
        <button class="mini-btn send" :disabled="store.isStreaming" @click="handleSend">🛩️</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.input-wrapper {
  position: absolute; bottom: 25px; left: 0; width: 100%; z-index: 999;
}
.input-card {
  background: #f0f4f9; border-radius: 24px;
  padding: 10px 15px; box-shadow: 0 4px 15px rgba(0,0,0,.05);
  overflow: visible;
}
.msg-input {
  width: 100%; border: none; background: transparent; resize: none;
  font-size: 15px; outline: none; max-height: 160px; overflow-y: auto;
  line-height: 1.5; padding: 4px 0;
}
.toolbar {
  display: flex; align-items: center; gap: 6px;
  flex-wrap: nowrap; margin-top: 6px;
}
/* anchor 提供相对定位基点，popup 向上弹出 */
.anchor { position: relative; flex-shrink: 0; }
.popup-panel {
  position: absolute; bottom: 100%; left: 0; margin-bottom: 8px;
  background: #fff; border-radius: 12px; padding: 12px;
  box-shadow: 0 -4px 20px rgba(0,0,0,.12);
  min-width: 160px; max-height: 260px; overflow-y: auto;
  z-index: 2000;
}
.panel-label { font-size: 13px; font-weight: 600; margin-bottom: 8px; color: #374151; }
.radio-item {
  padding: 6px 10px; border-radius: 6px; cursor: pointer;
  font-size: 13px; color: #374151;
}
.radio-item:hover { background: #f3f4f6; }
.radio-item.active { background: #bfdbfe; color: #1e40af; font-weight: 500; }
.pstatus { font-size: 12px; color: #6b7280; margin: 6px 0; }
.num-input {
  width: 100%; padding: 6px 8px; border: 1px solid #d1d5db;
  border-radius: 6px; font-size: 13px;
}
.mini-btn {
  border: none; background: #e5e7eb; border-radius: 6px;
  padding: 5px 12px; cursor: pointer; font-size: 13px; white-space: nowrap;
}
.mini-btn:hover    { background: #d1d5db; }
.mini-btn.active   { background: #3b82f6; color: #fff; }
.btn-sm {
  border: none; background: #e5e7eb; border-radius: 6px;
  padding: 4px 10px; cursor: pointer; font-size: 12px;
}
.btn-sm.primary { background: #3b82f6; color: #fff; }
.mini-btn.send { background: #3b82f6; color: #fff; font-size: 16px; }
.mini-btn.send:disabled { opacity: .5; cursor: not-allowed; }
</style>