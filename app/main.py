import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
# 确保你的 rag.py 路径正确，并且 final_chain 是可导入的
from app.services.rag import final_chain, store 

app = FastAPI(title="NJUST 校史助手 API")

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_user"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """流式返回 AI 的回答"""
    async def generate():
        # 调用你之前写好的 final_chain
        async for chunk in final_chain.astream(
            {"input": request.query},
            config={"configurable": {"session_id": request.session_id}}
        ):
            if "answer" in chunk:
                # 直接发送文本块
                yield chunk["answer"]

    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/clear")
async def clear_history(request: ChatRequest):
    """清空该 Session 的记忆"""
    if request.session_id in store:
        del store[request.session_id]
    return {"status": "cleared"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)