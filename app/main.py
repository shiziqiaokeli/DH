#用于构建API
from fastapi import FastAPI, HTTPException,UploadFile, File,Form
#用于调用异步生成器
from app.services.rag import RAGService
#流式输出响应
from fastapi.responses import StreamingResponse
#导入存储历史记录的容器
from redis import asyncio as aioredis
#导入会话模型
from app.core.schemas import ChatRequest, SessionItem
#导入配置文件
from app.core.config import settings
#导入获取会话历史函数
from app.services.rag import get_session_history
from langchain_core.messages import HumanMessage, AIMessage
import tempfile
import os
from app.services.rag import process_uploaded_file
from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.db.models import SystemSetting, KnowledgeBase, Prompt


#实例化FastAPI对象
app = FastAPI()
#实例化RAGService对象，让其初始化self参数
rag_service = RAGService()
#实例化Redis客户端
redis_client = aioredis.from_url(settings.REDIS_URL)
# 会话与自定义标题一一对应：单 HASH，field=session_id，value=标题（空串表示未设置）
SESSION_INDEX_KEY = "dh:sessions"


def _decode_str(raw: bytes | str | None) -> str | None:
    if raw is None:
        return None
    text = raw.decode() if isinstance(raw, bytes) else raw
    t = text.strip()
    return t if t else None


async def register_session_id(session_id: str) -> None:
    """首次对话时登记会话；已有记录不覆盖标题。"""
    exists = await redis_client.hexists(SESSION_INDEX_KEY, session_id)
    if not exists:
        await redis_client.hset(SESSION_INDEX_KEY, session_id, "")


async def unregister_session_id(session_id: str) -> None:
    """删除会话在索引中的整条记录（含标题）。"""
    await redis_client.hdel(SESSION_INDEX_KEY, session_id)


async def list_session_ids() -> list[str]:
    raw = await redis_client.hkeys(SESSION_INDEX_KEY)
    return sorted(r.decode() if isinstance(r, bytes) else r for r in raw)


async def get_session_title(session_id: str) -> str | None:
    raw = await redis_client.hget(SESSION_INDEX_KEY, session_id)
    return _decode_str(raw)


async def set_session_title(session_id: str, title: str) -> None:
    t = title.strip()
    await redis_client.hset(SESSION_INDEX_KEY, session_id, t)

async def get_active_collection() -> str:
    """从 MySQL system_settings → knowledge_bases 读取当前激活的 collection_name"""
    async with AsyncSessionLocal() as session:
        stmt = (
            select(KnowledgeBase.collection_name)
            .join(SystemSetting, SystemSetting.active_kb_id == KnowledgeBase.id)
            .where(SystemSetting.id == 1)
        )
        result = await session.execute(stmt)
        name = result.scalar_one_or_none()
        if name is None:
            raise HTTPException(status_code=500, detail="未配置活跃知识库")
        return name

async def get_active_qa_system_prompt() -> tuple[str, int]:
    """当前活跃提示词正文 + id，用于构建 chain 与缓存键。"""
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

async def get_active_t_is_voice() -> tuple[float, bool]:
    """读取 id=1 的 t_value（温度）、is_voice_mode。"""
    async with AsyncSessionLocal() as session:
        stmt = select(SystemSetting.t_value, SystemSetting.is_voice_mode).where(
            SystemSetting.id == 1
        )
        row = (await session.execute(stmt)).one_or_none()
        if row is None:
            raise HTTPException(status_code=500, detail="未配置 system_settings")
        t_val, voice = row[0], row[1]
        return float(t_val), bool(voice)

#定义聊天接口
@app.post("/chat")
async def text_text_chat(request: ChatRequest):
    await register_session_id(request.session_id)
    collection_name = await get_active_collection()  # 从 MySQL 读
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


#定义清除历史记录接口
@app.post("/delete")
async def clear_history(request: ChatRequest):
    #删除对应session_id的会话历史
    h = get_session_history(request.session_id)
    await h.clear()
    await unregister_session_id(request.session_id)


#定义获取会话历史接口
@app.get("/history/{session_id}")
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


@app.put("/sessions/{session_id}/title")
async def update_session_title(session_id: str, body: SessionItem) -> dict[str, bool]:
    """请求体为 SessionItem：title 为人读展示名，空串则清除 Redis 中的自定义标题"""
    if body.session_id != session_id:
        raise HTTPException(status_code=400, detail="session_id 与路径不一致")
    await set_session_title(session_id, body.title or "")
    return {"ok": True}


#定义获取会话列表接口
@app.get("/sessions", response_model=list[SessionItem])
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


@app.post("/rag/upload")
async def upload_rag_file(kb_name: str=Form(...), file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="目前仅支持 .txt 文件")
    
    # 第一步：处理文件，获得自动生成的 collection_name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        collection_name = await process_uploaded_file(tmp_path)
    finally:
        os.remove(tmp_path)
    
    # 第二步：将 name + collection_name 写入 MySQL
    async with AsyncSessionLocal() as session:
        kb = KnowledgeBase(name=kb_name, collection_name=collection_name)
        session.add(kb)
        await session.commit()
        await session.refresh(kb)
    
    return {"id": kb.id, "name": kb.name, "collection_name": collection_name}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)