#用于构建API
from fastapi import FastAPI
#用于调用异步生成器
from app.services.rag import RAGService
#导入请求模型
from app.core.schemas import ChatRequest
#流式输出响应
from fastapi.responses import StreamingResponse
#导入存储历史记录的容器
from redis import asyncio as aioredis
#导入配置文件
from app.core.config import settings
#导入获取会话历史函数
from app.services.rag import get_session_history
from langchain_core.messages import HumanMessage, AIMessage

#实例化FastAPI对象
app = FastAPI()
#实例化RAGService对象，让其初始化self参数
rag_service = RAGService()
#定义聊天接口
@app.post("/chat")
async def text_text_chat(request: ChatRequest):
    #给生成器传入请求的query和session_id
    async def generate():
        async for chunk in rag_service.chat(request.query, session_id=request.session_id):
            if chunk:
                yield chunk           
    #SSE响应格式，流式输出
    return StreamingResponse(generate(), media_type="text/event-stream")
#实例化Redis客户端
redis_client = aioredis.from_url(settings.REDIS_URL)
#定义清除历史记录接口
@app.post("/delete")
async def clear_history(request: ChatRequest):
    #删除对应session_id的会话历史
    h = get_session_history(request.session_id)
    await h.clear()
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)