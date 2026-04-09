<script setup>
import { ref } from 'vue'
import { useChatStore } from '../stores/chat.js'
import { api } from '../api/index.js'

const store = useChatStore()

// ── 知识库上传 ─────────────────────────────────────────
const kbInput    = ref(null)
const kbFile     = ref(null)
const kbName     = ref('')
const kbStatus   = ref('')
const showKbForm = ref(false)

function onKbFile(e) {
  kbFile.value = e.target.files[0]
  if (kbFile.value) { showKbForm.value = true; kbStatus.value = `已选择：${kbFile.value.name}` }
}
async function submitKb() {
  if (!kbName.value.trim()) { kbStatus.value = '⚠️ 请填写名称'; return }
  try {
    const r = await api.uploadKnowledgeBase(kbFile.value, kbName.value.trim())
    kbStatus.value = `✅ 知识库「${r.name}」创建成功`
    kbFile.value = null; kbName.value = ''; showKbForm.value = false
  } catch (e) { kbStatus.value = `❌ ${e.message}` }
}

// ── 提示词创建 ─────────────────────────────────────────
const defaultPromptBody = `你是一个/名XXXX的XXXX。 要求：
1. 仅根据资料内容回答，不要胡编乱造。
2. 如果资料中没提到相关信息，请尝试根据你已有的知识回答，并注明'根据通用知识补充'。
3. 回答语气要严谨
4. 回答要简洁明了。`
const promptBody   = ref(defaultPromptBody)
const promptName   = ref('')
const promptStatus = ref('')
const showPForm    = ref(false)

async function submitPrompt() {
  if (!promptBody.value.trim()) { promptStatus.value = '⚠️ 请填写内容'; return }
  if (!promptName.value.trim()) { promptStatus.value = '⚠️ 请填写名称'; return }
  try {
    const r = await api.createPrompt(promptName.value.trim(), promptBody.value.trim())
    promptStatus.value = `✅ 提示词「${r.name}」保存成功`
    promptName.value = ''; showPForm.value = false
  } catch (e) { promptStatus.value = `❌ ${e.message}` }
}

// ── GSV 模型训练 ───────────────────────────────────────
const modelInput    = ref(null)
const modelFile     = ref(null)
const modelName     = ref('')
const modelStatus   = ref('')
const showModelForm = ref(false)

function onModelFile(e) {
  modelFile.value = e.target.files[0]
  if (modelFile.value) { showModelForm.value = true; modelStatus.value = `已选择：${modelFile.value.name}` }
}
async function submitModel() {
  if (!modelName.value.trim()) { modelStatus.value = '⚠️ 请填写名称'; return }
  const exp = `model_${crypto.randomUUID().replace(/-/g, '').slice(0, 8)}`
  try {
    const r = await api.trainVoiceModel(modelFile.value, modelName.value.trim(), exp)
    modelStatus.value = `✅ 训练已提交，task_id: ${r.task_id}`
    modelFile.value = null; modelName.value = ''; showModelForm.value = false
  } catch (e) { modelStatus.value = `❌ ${e.message}` }
}

// ── 参考音频上传 ───────────────────────────────────────
const audioInput    = ref(null)
const audioFile     = ref(null)
const audioRefText  = ref('')
const audioName     = ref('')
const audioStatus   = ref('')
const showAudioForm = ref(false)

function onAudioFile(e) {
  audioFile.value = e.target.files[0]
  if (audioFile.value) { showAudioForm.value = true; audioStatus.value = `已选择：${audioFile.value.name}` }
}
async function submitAudio() {
  if (!audioRefText.value.trim()) { audioStatus.value = '⚠️ 请填写参考文本'; return }
  if (!audioName.value.trim())    { audioStatus.value = '⚠️ 请填写名称'; return }
  try {
    await api.uploadReferAudio(audioFile.value, audioName.value.trim(), audioRefText.value.trim())
    audioStatus.value = `✅ 参考音频「${audioName.value.trim()}」上传成功`
    audioFile.value = null; audioRefText.value = ''; audioName.value = ''; showAudioForm.value = false
  } catch (e) { audioStatus.value = `❌ ${e.message}` }
}
</script>

<template>
  <aside class="sidebar">
    <!-- 发起新对话 -->
    <button class="btn primary lg" @click="store.newChat">📝 发起新对话</button>

    <!-- 新建知识库 -->
    <input ref="kbInput" type="file" accept=".txt" hidden @change="onKbFile" />
    <button class="btn secondary lg" @click="kbInput.click()">新建知识库</button>
    <template v-if="showKbForm">
      <input v-model="kbName" class="field" placeholder="给知识库起个名字" />
      <button class="btn secondary lg" @click="submitKb">确认上传</button>
    </template>
    <p v-if="kbStatus" class="status">{{ kbStatus }}</p>

    <!-- 新建提示词 -->
    <button class="btn secondary lg" @click="showPForm = !showPForm">新建提示词</button>
    <template v-if="showPForm">
      <textarea v-model="promptBody" class="field" rows="6" />
      <input v-model="promptName" class="field" placeholder="给提示词起个名字" />
      <button class="btn secondary lg" @click="submitPrompt">确认保存</button>
    </template>
    <p v-if="promptStatus" class="status">{{ promptStatus }}</p>

    <!-- 训练 GSV 模型 -->
    <input ref="modelInput" type="file" accept=".wav,.mp3,.flac,.m4a" hidden @change="onModelFile" />
    <button class="btn secondary lg" @click="modelInput.click()">训练GSV模型</button>
    <template v-if="showModelForm">
      <input v-model="modelName" class="field" placeholder="给模型起个名字" />
      <button class="btn secondary lg" @click="submitModel">开始训练</button>
    </template>
    <p v-if="modelStatus" class="status">{{ modelStatus }}</p>

    <!-- 上传参考音频 -->
    <input ref="audioInput" type="file" accept=".wav,.mp3,.flac,.m4a" hidden @change="onAudioFile" />
    <button class="btn secondary lg" @click="audioInput.click()">上传参考音频</button>
    <template v-if="showAudioForm">
      <input v-model="audioRefText" class="field" placeholder="输入参考文本（与音频内容一致）" />
      <input v-model="audioName"    class="field" placeholder="给参考音频起个名字" />
      <button class="btn secondary lg" @click="submitAudio">确认上传</button>
    </template>
    <p v-if="audioStatus" class="status">{{ audioStatus }}</p>

    <!-- 会话列表 -->
    <div class="divider">对话</div>
    <nav class="session-list">
      <div
        v-for="s in store.sessions"
        :key="s.session_id"
        class="session-item"
        :class="{ active: s.session_id === store.sessionId }"
        @click="store.switchSession(s.session_id)"
      >{{ store.getDisplayName(s.session_id) }}</div>
    </nav>
  </aside>
</template>

<style scoped>
.sidebar {
  width: 220px; flex-shrink: 0; background: #f0f4f9;
  padding: 20px 15px; height: 100vh; overflow-y: auto;
  display: flex; flex-direction: column; gap: 8px;
}
.btn {
  width: 100%; border: none; border-radius: 8px;
  cursor: pointer; font-size: 14px; transition: background .15s;
}
.btn.primary    { background: #3b82f6; color: #fff; }
.btn.primary:hover { background: #2563eb; }
.btn.secondary  { background: #e5e7eb; color: #374151; }
.btn.secondary:hover { background: #d1d5db; }
.btn.lg { padding: 10px 12px; }
.field {
  width: 100%; padding: 8px 10px; border: 1px solid #d1d5db;
  border-radius: 8px; font-size: 13px; resize: vertical;
}
.status { font-size: 12px; color: #6b7280; word-break: break-all; }
.divider {
  font-size: 16px; font-weight: 600; text-align: center;
  margin-top: 8px; padding: 8px 0 4px; border-top: 1px solid #d1d5db;
}
.session-list { display: flex; flex-direction: column; gap: 4px; }
.session-item {
  padding: 8px 10px; border-radius: 8px; font-size: 13px;
  cursor: pointer; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.session-item:hover  { background: #e2e8f0; }
.session-item.active { background: #bfdbfe; color: #1e40af; font-weight: 500; }
</style>