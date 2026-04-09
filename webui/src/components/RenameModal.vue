<script setup>
import { ref } from 'vue'
import { useChatStore } from '../stores/chat.js'

const emit = defineEmits(['close'])
const store = useChatStore()
const sid = store.sessionId
const newName = ref(store.displayNames[sid] || sid || '')

async function confirm() {
  await store.renameSession(newName.value)
  emit('close')
}
</script>

<template>
  <div class="overlay" @click.self="$emit('close')">
    <div class="card">
      <div class="title">重命名会话</div>
      <input v-model="newName" class="input" placeholder="输入新名称" @keydown.enter="confirm" />
      <div class="btns">
        <button class="btn primary" @click="confirm">确认</button>
        <button class="btn" @click="$emit('close')">取消</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,.5);
  display: flex; align-items: center; justify-content: center; z-index: 9999;
}
.card {
  background: #fff; border-radius: 16px; padding: 24px 32px;
  width: 400px; max-width: 90vw; box-shadow: 0 8px 32px rgba(0,0,0,.2);
}
.title { font-size: 18px; font-weight: 600; text-align: center; margin-bottom: 16px; }
.input {
  width: 100%; padding: 12px 16px; border: 1px solid #d1d5db;
  border-radius: 8px; font-size: 14px; margin-bottom: 20px;
}
.btns { display: flex; justify-content: center; gap: 12px; }
.btn {
  border: none; border-radius: 8px; padding: 8px 24px;
  cursor: pointer; font-size: 14px; background: #e5e7eb;
}
.btn.primary { background: #3b82f6; color: #fff; }
</style>