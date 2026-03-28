#用于构建API
from fastapi import FastAPI
#用于调用异步生成器
from app.services.rag import RAGService
#导入请求模型
from app.core.schemas import ChatRequest
#流式输出响应
from fastapi.responses import StreamingResponse
#导入存储历史记录的容器
from app.services.rag import store

#实例化FastAPI对象
app = FastAPI()
#实例化RAGService对象，让其初始化self参数
rag_service = RAGService()
#定义聊天接口
@app.post("/chat")
async def text_text_chat(request: ChatRequest):
    #给生成器传入请求的query和session_id
    gen = rag_service.chat(request.query, request.session_id)
    #SSE响应格式，流式输出
    return StreamingResponse(gen, media_type="text/event-stream")
#定义清除历史记录接口
@app.post("/clear")
async def clear_history(request: ChatRequest):
    #通过检索session_id来删除对应的历史记录
    if request.session_id in store:
        del store[request.session_id]
    return {"status": "cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)