from fastapi import FastAPI,HTTPException,UploadFile,File,Form
from app.services.rag import RAGService,get_session_history,process_uploaded_file
from redis import asyncio as aioredis
from app.core.config import settings
from app.db.session import AsyncSessionLocal
from sqlalchemy import select
from app.db.models import SystemSetting,KnowledgeBase,Prompt,ReferAudio,VoiceModel
from app.core.schemas import ChatRequest,SessionItem
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage,AIMessage
import tempfile
import os
import httpx

app = FastAPI()
rag_service = RAGService()   
redis_client = aioredis.from_url(settings.REDIS_URL)
SESSION_INDEX_KEY = "dh:sessions"   

def _decode_str(raw: bytes | str | None) -> str | None:   #清洗和标准化从Redis或其他数据源获取的字符串数据（因为Redis返回的是bytes，所以要解码）
    if raw is None:
        return None
    text = raw.decode() if isinstance(raw, bytes) else raw
    t = text.strip()
    return t if t else None

async def register_session_id(session_id: str) -> None:   #首次对话时登记会话；已有记录无事发生
    exists = await redis_client.hexists(SESSION_INDEX_KEY, session_id)   #通过session_id检查会话是否存在
    if not exists:
        await redis_client.hset(SESSION_INDEX_KEY, session_id, "")   #如果会话不存在，则创建会话

async def unregister_session_id(session_id: str) -> None:   #删除会话在索引中的整条记录（含标题）
    await redis_client.hdel(SESSION_INDEX_KEY, session_id)

async def list_session_ids() -> list[str]:   #列出所有会话ID
    raw = await redis_client.hkeys(SESSION_INDEX_KEY)
    return sorted(r.decode() if isinstance(r, bytes) else r for r in raw)

async def get_session_title(session_id: str) -> str | None:   #获取会话标题
    raw = await redis_client.hget(SESSION_INDEX_KEY, session_id)
    return _decode_str(raw)

async def set_session_title(session_id: str, title: str) -> None:   #设置会话标题
    t = title.strip()
    await redis_client.hset(SESSION_INDEX_KEY, session_id, t)

async def get_active_collection() -> str:   #通过system_settings的active_kb_id连接knowledge_bases读取当前激活的collection_name
    async with AsyncSessionLocal() as session:
        stmt = (
            select(KnowledgeBase.collection_name)   #终点
            .join(SystemSetting, SystemSetting.active_kb_id == KnowledgeBase.id)   #桥梁
            .where(SystemSetting.id == 1)   #起点
        )
        result = await session.execute(stmt)
        name = result.scalar_one_or_none()
        if name is None:
            raise HTTPException(status_code=500, detail="未配置活跃知识库")
        return name

async def get_active_qa_system_prompt() -> tuple[str, int]:   #通过system_settings的active_prompt_id连接prompts读取当前激活的prompt_content和prompt_id
    async with AsyncSessionLocal() as session:
        stmt = (
            select(Prompt.content, Prompt.id)
            .join(SystemSetting, SystemSetting.active_prompt_id == Prompt.id)
            .where(SystemSetting.id == 1)
        )
        row = (await session.execute(stmt)).one_or_none()
        if row is None:
            raise HTTPException(status_code=500, detail="未配置活跃提示词")
        content, pid = row[0], row[1]
        content = content.rstrip() + "\n\n请根据提供的【参考资料】来回答问题。【参考资料】：{context}"
        return content, pid

async def get_active_refer_audio() -> ReferAudio:   #通过system_settings的active_audio_id连接refer_audios读取当前激活的refer_audio
    async with AsyncSessionLocal() as session:
        stmt = (
            select(ReferAudio)
            .join(SystemSetting, SystemSetting.active_audio_id == ReferAudio.id)
            .where(SystemSetting.id == 1)
        )
        audio = (await session.execute(stmt)).scalar_one_or_none()
        if audio is None:
            raise HTTPException(status_code=500, detail="未配置活跃参考音频")
        return audio

async def get_active_t_is_voice() -> tuple[float, bool]:   #通过system_settings的id=1读取t_value和is_voice_mode
    async with AsyncSessionLocal() as session:
        stmt = select(SystemSetting.t_value, SystemSetting.is_voice_mode).where(
            SystemSetting.id == 1
        )
        row = (await session.execute(stmt)).one_or_none()
        if row is None:
            raise HTTPException(status_code=500, detail="未配置 system_settings")
        t_val, voice = row[0], row[1]
        return float(t_val), bool(voice)

@app.post("/chat")   #核心
async def text_text_chat(request: ChatRequest):
    await register_session_id(request.session_id)
    collection_name = await get_active_collection()  
    qa_system_prompt, prompt_id = await get_active_qa_system_prompt()
    temperature, is_voice_mode = await get_active_t_is_voice()
    async def generate():
        async for chunk in rag_service.chat(
            request.query,
            session_id=request.session_id,
            collection_name=collection_name, 
            qa_system_prompt=qa_system_prompt,
            prompt_id=prompt_id,
            temperature=temperature,
            is_voice_mode=is_voice_mode,
        ):
            if chunk:
                yield chunk
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/history/{session_id}")   #获取会话历史
async def get_chat_history(session_id: str):
    history_store = rag_service.get_history(session_id)
    messages = history_store.messages
    out = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            out.append({"role": "assistant", "content": m.content})
    return out

@app.put("/sessions/{session_id}/title")  #设置会话标题
async def update_session_title(session_id: str, body: SessionItem) -> dict[str, bool]:
    if body.session_id != session_id:
        raise HTTPException(status_code=400, detail="session_id 与路径不一致")
    await set_session_title(session_id, body.title or "")
    return {"ok": True}

@app.post("/delete")   #清除会话历史
async def clear_history(request: ChatRequest):
    h = get_session_history(request.session_id)
    await h.clear()   #删除对应session_id的会话历史
    await unregister_session_id(request.session_id)   #删除会话在索引中的整条记录（含标题）

@app.get("/sessions", response_model=list[SessionItem])   #获取会话列表
async def list_sessions():
    ids = await list_session_ids()
    out: list[SessionItem] = []
    for sid in ids:
        h = get_session_history(sid)
        try:
            msgs = h.messages
        except Exception:
            msgs = []
        if msgs:
            t = await get_session_title(sid)
            out.append(SessionItem(session_id=sid, title=t))
        else:
            await unregister_session_id(sid)
    return out

@app.post("/rag/upload")   #上传文件到知识库
async def upload_rag_file(kb_name: str=Form(...), file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="目前仅支持 .txt 文件")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:   #处理文件，获得自动生成的collection_name
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        collection_name = await process_uploaded_file(tmp_path)
    finally:
        os.remove(tmp_path)
    async with AsyncSessionLocal() as session:   #将name+collection_name写入 MySQL
        kb = KnowledgeBase(name=kb_name, collection_name=collection_name)
        session.add(kb)
        await session.commit()
        await session.refresh(kb)
    return {"id": kb.id, "name": kb.name, "collection_name": collection_name}

@app.get("/knowledgebases")   #列出所有知识库，并返回当前激活的active_kb_id
async def list_knowledgebases(): 
    async with AsyncSessionLocal() as session:
        kbs = (await session.execute(select(KnowledgeBase))).scalars().all()
        setting = (
            await session.execute(
                select(SystemSetting).where(SystemSetting.id == 1)
            )
        ).scalar_one_or_none()
        active_kb_id = setting.active_kb_id if setting else None
    return {
        "active_kb_id": active_kb_id,
        "knowledgebases": [{"id": kb.id, "name": kb.name} for kb in kbs],
    }

@app.put("/settings/active_kb")   #更新system_settings中的active_kb_id
async def update_active_kb(body: dict):
    kb_id = body.get("kb_id")
    if not kb_id:
        raise HTTPException(status_code=400, detail="缺少 kb_id")
    async with AsyncSessionLocal() as session:
        setting = (
            await session.execute(
                select(SystemSetting).where(SystemSetting.id == 1)
            )
        ).scalar_one_or_none()
        if setting is None:
            raise HTTPException(status_code=500, detail="system_settings 未初始化")
        setting.active_kb_id = int(kb_id)
        await session.commit()
    return {"ok": True, "active_kb_id": kb_id}

@app.post("/prompts")   #新建一条提示词
async def create_prompt(
    prompt_name: str = Form(..., description="提示词显示名称"),
    prompt_body: str = Form(..., description="提示词正文内容"),
):
    if not prompt_name or not prompt_name.strip():
        raise HTTPException(status_code=400, detail="prompt_name 不能为空")
    if not prompt_body or not prompt_body.strip():
        raise HTTPException(status_code=400, detail="prompt_body 不能为空")
    async with AsyncSessionLocal() as session:
        exists = (
            await session.execute(
                select(Prompt).where(Prompt.name == prompt_name.strip())
            )
        ).scalar_one_or_none()
        if exists:
            raise HTTPException(status_code=409, detail=f"名称「{prompt_name.strip()}」已存在")
        p = Prompt(name=prompt_name.strip(), content=prompt_body.strip())
        session.add(p)
        await session.commit()
        await session.refresh(p)
    return {"id": p.id, "name": p.name}

@app.get("/prompts")   #列出所有提示词，并返回当前激活的active_prompt_id
async def list_prompts():
    async with AsyncSessionLocal() as session:
        pbs = (await session.execute(select(Prompt))).scalars().all()
        setting = (
            await session.execute(
                select(SystemSetting).where(SystemSetting.id == 1)
            )
        ).scalar_one_or_none()
        active_prompt_id = setting.active_prompt_id if setting else None
    return {
        "active_prompt_id": active_prompt_id,
        "prompts": [{"id": pb.id, "name": pb.name} for pb in pbs],
    }

@app.put("/settings/active_prompt")   #更新system_settings中的active_prompt_id
async def update_active_pb(body: dict):
    prompt_id = body.get("prompt_id")
    if not prompt_id:
        raise HTTPException(status_code=400, detail="缺少 pb_id")
    async with AsyncSessionLocal() as session:
        setting = (
            await session.execute(
                select(SystemSetting).where(SystemSetting.id == 1)
            )
        ).scalar_one_or_none()
        if setting is None:
            raise HTTPException(status_code=500, detail="system_settings 未初始化")
        setting.active_prompt_id = int(prompt_id)
        await session.commit()
    return {"ok": True, "active_prompt_id": prompt_id}

@app.post("/voice_models/train")   #训练语音模型(接收前端上传的音频和模型名称，转发到GPT-SoVITS/train/start_with_asr，立即返回task_id)
async def start_voice_model_train(   
    model_name: str = Form(..., description="模型在系统中的显示名称"),
    audio_file: UploadFile = File(..., description="训练用原始音频"),
    exp_name:   str = Form(..., description="实验名（英文+下划线）"),
):
    gsv_base = settings.TRAIN_URL
    audio_bytes = await audio_file.read()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{gsv_base}/train/start_with_asr",
                data={   #训练参数
                    "exp_name":   exp_name.strip().replace(" ", "_"),
                    "version":    "v2Pro",
                    "gpu_numbers": "0",
                    "asr_model":  "Faster Whisper (多语种)",
                    "asr_model_size": "large-v3",
                    "asr_lang":   "zh",
                    "sovits_epoch": 4,
                    "sovits_save_every": 4,
                    "gpt_epoch":  10,
                    "gpt_save_every": 10,
                    "batch_size": 2,
                    "if_save_latest": True,
                    "if_save_every_weights": False,
                },
                files={"audio_file": (audio_file.filename, audio_bytes, audio_file.content_type)},
            )
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="无法连接到 GPT-SoVITS 训练服务")
    if resp.status_code not in (200, 202):
        raise HTTPException(status_code=502, detail=f"训练服务拒绝请求: {resp.text}")
    task_id = resp.json().get("task_id")
    await redis_client.hset(   #将task_id→model_name+exp_name的映射存入Redis，供轮询时使用
        "dh:train_tasks",
        task_id,
        f"{model_name}||{exp_name.strip().replace(' ', '_')}",
    )
    return {"task_id": task_id, "message": "训练任务已提交"}

@app.get("/voice_models/train/status/{task_id}")   #训练结果需通过8000/voice_models/train/status/{task_id}轮询    
async def poll_train_status(task_id: str):   
    gsv_base = settings.TRAIN_URL
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{gsv_base}/train/status/{task_id}")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="无法连接到 GPT-SoVITS 训练服务")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=resp.text)
    data = resp.json()
    status   = data.get("status")   #"pending"/"running"/"done"/"error"
    gpt_path    = data.get("gpt_path")
    sovits_path = data.get("sovits_path")
    if status == "done" and gpt_path and sovits_path:   #训练完成，且路径都有值时，自动写入MySQL（只写一次）
        meta_raw = await redis_client.hget("dh:train_tasks", task_id)
        if meta_raw:
            meta = (meta_raw.decode() if isinstance(meta_raw, bytes) else meta_raw)
            model_name, exp_name = meta.split("||", 1)
            async with AsyncSessionLocal() as session:
                exists = (await session.execute(   #防重复写入
                    select(VoiceModel).where(VoiceModel.name == model_name)
                )).scalar_one_or_none()
                if not exists:
                    session.add(VoiceModel(
                        name=model_name,
                        pth_path=sovits_path,
                        ckpt_path=gpt_path,
                    ))
                    await session.commit()
            await redis_client.hdel("dh:train_tasks", task_id)   #写完后删掉Redis里的临时记录
    return data   #原样透传GPT-SoVITS返回的完整状态

@app.get("/voice_models")   #列出所有语音模型，并返回当前激活的active_model_id
async def list_voice_models():
    async with AsyncSessionLocal() as session:
        models = (await session.execute(select(VoiceModel))).scalars().all()
        setting = (
            await session.execute(
                select(SystemSetting).where(SystemSetting.id == 1)
            )
        ).scalar_one_or_none()
        active_model_id = setting.active_model_id if setting else None
    return {
        "active_model_id": active_model_id,
        "voice_models": [{"id": m.id, "name": m.name} for m in models],
    }

@app.put("/settings/active_model")   #更新system_settings中的active_model_id(真实的推理需要更改GSV项目里的推理路径)
async def update_active_model(body: dict):
    model_id = body.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="缺少 model_id")
    async with AsyncSessionLocal() as session:
        setting = (
            await session.execute(
                select(SystemSetting).where(SystemSetting.id == 1)
            )
        ).scalar_one_or_none()
        if setting is None:
            raise HTTPException(status_code=500, detail="system_settings 未初始化")
        setting.active_model_id = int(model_id)
        await session.commit()
        model = (
            await session.execute(
                select(VoiceModel).where(VoiceModel.id == int(model_id))
            )
        ).scalar_one_or_none()
        if model is None:
            raise HTTPException(status_code=404, detail="未找到该语音模型")
    gsv_base = settings.TTS_URL
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            gpt_r = await client.get(
                f"{gsv_base}/set_gpt_weights",
                params={"weights_path": model.ckpt_path},
            )
            sovits_r = await client.get(
                f"{gsv_base}/set_sovits_weights",
                params={"weights_path": model.pth_path},
            )
        if gpt_r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"GPT 权重切换失败: {gpt_r.text}")
        if sovits_r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"SoVITS 权重切换失败: {sovits_r.text}")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="无法连接到 GPT-SoVITS 服务")
    return {"ok": True, "active_model_id": model_id}

@app.post("/refer_audios")   #上传参考音频(转发到GPT-SoVITS/upload_refer_audio保存，取回绝对路径后写入本地refer_audios表)
async def create_refer_audio(
    name: str = Form(..., description="参考音频的显示名称"),
    ref_text: str = Form(..., description="与音频内容一致的参考文本"),
    audio_file: UploadFile = File(..., description="音频文件（wav/mp3/flac 等）"),
):
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="name 不能为空")
    if not ref_text or not ref_text.strip():
        raise HTTPException(status_code=400, detail="ref_text 不能为空")
    audio_bytes = await audio_file.read()
    gsv_base = settings.TTS_URL
    try:   
        async with httpx.AsyncClient(timeout=60.0) as client:   #转发到GPT-SoVITS，获取文件落盘的绝对路径
            resp = await client.post(
                f"{gsv_base}/upload_refer_audio",
                files={
                    "refer_audio_file": (
                        audio_file.filename,
                        audio_bytes,
                        audio_file.content_type or "audio/wav",
                    )
                },
            )
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="无法连接到 GPT-SoVITS 服务")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"GPT-SoVITS 上传失败: {resp.text}")
    audio_path = resp.json().get("path")
    if not audio_path:
        raise HTTPException(status_code=502, detail="GPT-SoVITS 未返回有效路径")
    async with AsyncSessionLocal() as session:    #写入本地refer_audios表
        exists = (   #防止name重复导致唯一索引冲突
            await session.execute(
                select(ReferAudio).where(ReferAudio.name == name.strip())
            )
        ).scalar_one_or_none()
        if exists:
            raise HTTPException(status_code=409, detail=f"名称「{name.strip()}」已存在")
        audio = ReferAudio(
            name=name.strip(),
            audio_path=audio_path,
            text=ref_text.strip(),
        )
        session.add(audio)
        await session.commit()
        await session.refresh(audio)
    return {"id": audio.id, "name": audio.name, "audio_path": audio.audio_path}

@app.get("/refer_audios")   #列出所有参考音频，并返回当前激活的active_audio_id
async def list_refer_audios():
    async with AsyncSessionLocal() as session:
        audios = (await session.execute(select(ReferAudio))).scalars().all()
        setting = (
            await session.execute(
                select(SystemSetting).where(SystemSetting.id == 1)
            )
        ).scalar_one_or_none()
        active_audio_id = setting.active_audio_id if setting else None
    return {
        "active_audio_id": active_audio_id,
        "refer_audios": [{"id": a.id, "name": a.name} for a in audios],
    }

@app.put("/settings/active_audio")   #更新system_settings中的active_audio_id
async def update_active_audio(body: dict):
    audio_id = body.get("audio_id")
    if not audio_id:
        raise HTTPException(status_code=400, detail="缺少 audio_id")
    async with AsyncSessionLocal() as session:
        setting = (
            await session.execute(
                select(SystemSetting).where(SystemSetting.id == 1)
            )
        ).scalar_one_or_none()
        if setting is None:
            raise HTTPException(status_code=500, detail="system_settings 未初始化")
        setting.active_audio_id = int(audio_id)
        await session.commit()
    return {"ok": True, "active_audio_id": audio_id}

@app.get("/settings/t_value")   #读取当前温度参数
async def get_t_value():
    async with AsyncSessionLocal() as session:
        setting = (
            await session.execute(
                select(SystemSetting).where(SystemSetting.id == 1)
            )
        ).scalar_one_or_none()
        if setting is None:
            raise HTTPException(status_code=500, detail="system_settings 未初始化")
    return {"t_value": float(setting.t_value)}

@app.put("/settings/t_value")   #更新温度参数，必须为正浮点数
async def update_t_value(body: dict):
    t_val = body.get("t_value")
    if t_val is None:
        raise HTTPException(status_code=400, detail="缺少 t_value")
    try:
        t_val = float(t_val)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="t_value 必须为数字")
    if t_val <= 0:
        raise HTTPException(status_code=400, detail="t_value 必须大于 0")
    async with AsyncSessionLocal() as session:
        setting = (
            await session.execute(
                select(SystemSetting).where(SystemSetting.id == 1)
            )
        ).scalar_one_or_none()
        if setting is None:
            raise HTTPException(status_code=500, detail="system_settings 未初始化")
        setting.t_value = t_val
        await session.commit()
    return {"ok": True, "t_value": t_val}

@app.get("/settings/voice_mode")   #读取当前是否为语音模式
async def get_voice_mode():
    async with AsyncSessionLocal() as session:
        setting = (
            await session.execute(
                select(SystemSetting).where(SystemSetting.id == 1)
            )
        ).scalar_one_or_none()
        if setting is None:
            raise HTTPException(status_code=500, detail="system_settings 未初始化")
    return {"is_voice_mode": bool(setting.is_voice_mode)}

@app.put("/settings/toggle_voice_mode")   #将is_voice_mode取反并保存，返回新状态
async def toggle_voice_mode():
    async with AsyncSessionLocal() as session:
        setting = (
            await session.execute(
                select(SystemSetting).where(SystemSetting.id == 1)
            )
        ).scalar_one_or_none()
        if setting is None:
            raise HTTPException(status_code=500, detail="system_settings 未初始化")
        setting.is_voice_mode = not setting.is_voice_mode
        new_val = setting.is_voice_mode
        await session.commit()
    return {"is_voice_mode": new_val}

@app.get("/tts_proxy")   #读取当前活跃参考音频配置，代理调用GPT-SoVITS，流式返回WAV音频
async def tts_proxy(text: str) -> StreamingResponse:
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")
    audio = await get_active_refer_audio()
    gsv_base = settings.TTS_URL
    params = {
        "text": text,
        "text_lang": "zh",
        "ref_audio_path": audio.audio_path,
        "prompt_lang": "zh",
        "prompt_text": audio.text,
        "media_type": "wav",
        "streaming_mode": True,   
        "speed_factor": 1.0,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
    }
    async def audio_generator():
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("GET", f"{gsv_base}/tts", params=params) as resp:
                    if resp.status_code != 200:
                        return
                    async for chunk in resp.aiter_bytes(chunk_size=4096):
                        if chunk:
                            yield chunk
        except httpx.ConnectError:
            pass
    return StreamingResponse(audio_generator(), media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)