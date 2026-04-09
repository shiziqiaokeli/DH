<script setup>
import { ref, computed, nextTick, watch } from 'vue'
import { marked } from 'marked'
import { useChatStore } from '../stores/chat.js'
import InputArea from './InputArea.vue'
import RenameModal from './RenameModal.vue'

const store = useChatStore()
const chatEl         = ref(null)
const showMoreMenu   = ref(false)
const showRenameModal = ref(false)

const currentTitle = computed(() =>
  store.sessionId ? (store.displayNames[store.sessionId] || store.sessionId) : 'default'
)

// 消息更新时自动滚到底部
watch(
  () => store.messages.at(-1)?.content,
  async () => { await nextTick(); chatEl.value?.scrollTo(0, chatEl.value.scrollHeight) }
)

function renderMd(text) {
  return marked.parse(text || '', { breaks: true })
}

function onDeleteSession() {
  showMoreMenu.value = false
  store.deleteCurrentSession()
}

// 点击其他区域关闭 more 菜单
function onAreaClick(e) {
  if (!e.target.closest('#more-anchor')) showMoreMenu.value = false
}
</script>

<template>
  <div class="chat-area" @click="onAreaClick">

    <!-- 顶部导航栏 -->
    <header class="top-nav">
      <div class="nav-item">人工智能机器人</div>
      <div class="nav-item nav-center">{{ currentTitle }}</div>
      <div class="nav-item nav-right">
        <div id="more-anchor" class="more-anchor">
          <button class="mini-btn" @click.stop="showMoreMenu = !showMoreMenu">更多</button>
          <div v-if="showMoreMenu" class="more-dropdown">
            <button class="mini-btn" @click="showRenameModal = true; showMoreMenu = false">重命名</button>
            <button class="mini-btn" @click="onDeleteSession">删除</button>
          </div>
        </div>
      </div>
    </header>

    <!-- 消息区 -->
    <div class="center-container">
      <div ref="chatEl" class="chat-window">

        <!-- 空状态占位 -->
        <div v-if="!store.messages.length" class="placeholder">
          <p style="font-size:24px">你好</p>
          <p style="font-size:30px; font-weight:700">需要我为你做些什么？</p>
        </div>

        <!-- 消息列表 -->
        <div v-for="(msg, i) in store.messages" :key="i" class="msg-row" :class="msg.role">
          <div class="bubble" v-html="renderMd(msg.content)"></div>
        </div>

        <!-- 打字中动画 -->
        <div v-if="store.isStreaming && !store.messages.at(-1)?.content" class="typing">
          <span /><span /><span />
        </div>
      </div>

      <!-- TTS 播放器（隐藏） -->
      <audio v-if="store.audioUrl" :src="store.audioUrl" autoplay style="display:none"
             @ended="store.audioUrl = null" />

      <InputArea />
    </div>

    <!-- 重命名弹窗 -->
    <RenameModal v-if="showRenameModal" @close="showRenameModal = false" />
  </div>
</template>

<style scoped>
.chat-area { flex: 1; position: relative; height: 100vh; display: flex; flex-direction: column; }
/* 顶部导航 */
.top-nav {
  position: absolute; top: 15px; left: 0; width: 100%;
  padding: 0 20px; display: flex; justify-content: space-between;
  align-items: center; z-index: 1000;
}
.nav-item    { flex: 1; font-size: 16px; font-weight: 600; }
.nav-center  { text-align: center; }
.nav-right   { display: flex; justify-content: flex-end; }
.more-anchor { position: relative; }
.more-dropdown {
  position: absolute; top: 100%; right: 0; margin-top: 6px;
  background: #fff; border-radius: 12px; padding: 8px;
  box-shadow: 0 4px 20px rgba(0,0,0,.12);
  display: flex; flex-direction: column; gap: 8px;
  min-width: 120px; z-index: 1001;
}
.mini-btn {
  border: none; background: #e5e7eb; border-radius: 6px;
  padding: 4px 12px; cursor: pointer; font-size: 13px; white-space: nowrap;
}
.mini-btn:hover { background: #d1d5db; }
/* 中央容器 */
.center-container {
  max-width: 850px; margin: 0 auto; height: 100vh;
  width: 100%; position: relative;
}
/* 消息区 */
.chat-window {
  height: 100%; overflow-y: auto;
  padding: 80px 20px 220px;
  display: flex; flex-direction: column; gap: 16px;
}
.placeholder { text-align: center; margin: auto; color: #9ca3af; }
.msg-row { display: flex; }
.msg-row.user     { justify-content: flex-end; }
.msg-row.assistant { justify-content: flex-start; }
.bubble {
  max-width: 70%; padding: 10px 14px; border-radius: 14px;
  font-size: 14px; line-height: 1.6; word-break: break-word;
}
.user .bubble     { background: #3b82f6; color: #fff; border-bottom-right-radius: 4px; }
.assistant .bubble { background: #f3f4f6; color: #1f2937; border-bottom-left-radius: 4px; }
/* 打字动画 */
.typing { display: flex; gap: 4px; padding: 10px 14px; align-self: flex-start; }
.typing span {
  width: 8px; height: 8px; border-radius: 50%; background: #9ca3af;
  animation: bounce .8s infinite alternate;
}
.typing span:nth-child(2) { animation-delay: .2s; }
.typing span:nth-child(3) { animation-delay: .4s; }
@keyframes bounce { to { transform: translateY(-6px); opacity: .4; } }
/* markdown 样式 */
.bubble :deep(pre)  { background: #1e293b; color: #e2e8f0; padding: 12px; border-radius: 8px; overflow-x: auto; }
.bubble :deep(code) { font-family: monospace; font-size: 13px; }
.bubble :deep(p)    { margin-bottom: 6px; }
.bubble :deep(ul), .bubble :deep(ol) { padding-left: 18px; margin-bottom: 6px; }
</style>