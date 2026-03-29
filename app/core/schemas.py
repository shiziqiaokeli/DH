from pydantic import BaseModel

#定义请求模型
class ChatRequest(BaseModel):
    query: str
    session_id: str 