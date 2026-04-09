// 所有后端接口封装（Vite proxy 后直接使用相对路径）

export const api = {
    // ── 流式对话 ──────────────────────────────────────────
    async *streamChat(query, sessionId) {
      const resp = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, session_id: sessionId }),
        signal: AbortSignal.timeout(60000),
      })
      if (!resp.ok) throw new Error(`后端错误 (${resp.status})`)
      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          const chunk = decoder.decode(value, { stream: true })
          if (chunk) yield chunk
        }
      } finally {
        reader.releaseLock()
      }
    },
  
    // ── 会话管理 ──────────────────────────────────────────
    getSessions:          ()        => fetch('/sessions').then(r => r.json()),
    getHistory:           (sid)     => fetch(`/history/${sid}`).then(r => r.json()),
    deleteSession:        (sid)     => fetch('/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: '', session_id: sid }),
    }),
    updateSessionTitle:   (sid, t)  => fetch(`/sessions/${sid}/title`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sid, title: t }),
    }),
  
    // ── 知识库 ────────────────────────────────────────────
    getKnowledgeBases: () => fetch('/knowledgebases').then(r => r.json()),
    uploadKnowledgeBase(file, kbName) {
      const fd = new FormData()
      fd.append('kb_name', kbName)
      fd.append('file', file)
      return fetch('/rag/upload', { method: 'POST', body: fd }).then(r => r.json())
    },
    setActiveKb: (id) => fetch('/settings/active_kb', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kb_id: id }),
    }),
  
    // ── 提示词 ────────────────────────────────────────────
    getPrompts: () => fetch('/prompts').then(r => r.json()),
    createPrompt(name, body) {
      const fd = new FormData()
      fd.append('prompt_name', name)
      fd.append('prompt_body', body)
      return fetch('/prompts', { method: 'POST', body: fd }).then(r => r.json())
    },
    setActivePrompt: (id) => fetch('/settings/active_prompt', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt_id: id }),
    }),
  
    // ── 语音模型 ──────────────────────────────────────────
    getVoiceModels: () => fetch('/voice_models').then(r => r.json()),
    trainVoiceModel(file, modelName, expName) {
      const fd = new FormData()
      fd.append('model_name', modelName)
      fd.append('exp_name', expName)
      fd.append('audio_file', file)
      return fetch('/voice_models/train', { method: 'POST', body: fd }).then(r => r.json())
    },
    setActiveModel: (id) => fetch('/settings/active_model', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: id }),
    }),
  
    // ── 参考音频 ──────────────────────────────────────────
    getReferAudios: () => fetch('/refer_audios').then(r => r.json()),
    uploadReferAudio(file, name, refText) {
      const fd = new FormData()
      fd.append('name', name)
      fd.append('ref_text', refText)
      fd.append('audio_file', file)
      return fetch('/refer_audios', { method: 'POST', body: fd }).then(r => r.json())
    },
    setActiveAudio: (id) => fetch('/settings/active_audio', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ audio_id: id }),
    }),
  
    // ── 温度 & 语音模式 ───────────────────────────────────
    getTemperature:   ()    => fetch('/settings/t_value').then(r => r.json()),
    setTemperature:   (val) => fetch('/settings/t_value', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ t_value: val }),
    }),
    getVoiceMode:     ()    => fetch('/settings/voice_mode').then(r => r.json()),
    toggleVoiceMode:  ()    => fetch('/settings/toggle_voice_mode', { method: 'PUT' }).then(r => r.json()),
  
    // ── TTS ───────────────────────────────────────────────
    getTtsUrl: (text) => `/tts_proxy?text=${encodeURIComponent(text)}`,
  }