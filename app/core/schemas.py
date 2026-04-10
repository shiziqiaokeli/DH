from typing import Annotated, Any
from pydantic import BaseModel, BeforeValidator

def extract_gradio_text(v: Any) -> str:
    if isinstance(v, list):   #只处理文本列表格式:[{'text':'...','type':'text'}]
        #提取列表中所有type为text的内容并拼接
        return "".join([str(item.get("text", "")) for item in v if isinstance(item, dict) and item.get("type") == "text"])
    return str(v) if v is not None else ""   #如果已经是字符串，直接返回
GradioString = Annotated[str, BeforeValidator(extract_gradio_text)]   #定义自适应字符串，通过前置校验器过滤非字符串内容
class ChatRequest(BaseModel):
    query: GradioString
    session_id: str
class SessionItem(BaseModel):
    session_id: str
    title: str | None = None