#用于构建API
from fastapi import FastAPI, HTTPException
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

#实例化FastAPI对象
app = FastAPI()
#实例化RAGService对象，让其初始化self参数
rag_service = RAGService()
#实例化Redis客户端
redis_client = aioredis.from_url(settings.REDIS_URL)
SESSION_SET_KEY = "dh:sessions"


async def register_session_id(session_id: str) -> None:
    await redis_client.sadd(SESSION_SET_KEY, session_id)


async def unregister_session_id(session_id: str) -> None:
    await redis_client.srem(SESSION_SET_KEY, session_id)


async def list_session_ids() -> list[str]:
    raw = await redis_client.smembers(SESSION_SET_KEY)
    return sorted(s.decode() if isinstance(s, bytes) else s for s in raw)


def _session_title_key(session_id: str) -> str:
    return f"dh:session:title:{session_id}"


async def get_session_title(session_id: str) -> str | None:
    raw = await redis_client.get(_session_title_key(session_id))
    if raw is None:
        return None
    return raw.decode() if isinstance(raw, bytes) else raw


async def set_session_title(session_id: str, title: str) -> None:
    t = title.strip()
    if not t:
        await redis_client.delete(_session_title_key(session_id))
    else:
        await redis_client.set(_session_title_key(session_id), t)


async def delete_session_title(session_id: str) -> None:
    await redis_client.delete(_session_title_key(session_id))


#定义聊天接口
@app.post("/chat")
async def text_text_chat(request: ChatRequest):
    await register_session_id(request.session_id)
    #给生成器传入请求的query和session_id
    async def generate():
        async for chunk in rag_service.chat(request.query, session_id=request.session_id):
            if chunk:
                yield chunk
    #SSE响应格式，流式输出
    return StreamingResponse(generate(), media_type="text/event-stream")


#定义清除历史记录接口
@app.post("/delete")
async def clear_history(request: ChatRequest):
    #删除对应session_id的会话历史
    h = get_session_history(request.session_id)
    await h.clear()
    await unregister_session_id(request.session_id)
    await delete_session_title(request.session_id)


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)