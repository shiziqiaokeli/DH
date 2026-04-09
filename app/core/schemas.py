from typing import Annotated, Any, List, Dict
from pydantic import BaseModel, BeforeValidator

def extract_gradio_text(v: Any) -> str:
    #只处理文本列表格式:[{'text':'...','type':'text'}]
    if isinstance(v, list):
        #提取列表中所有type为text的内容并拼接
        return "".join([str(item.get("text", "")) for item in v if isinstance(item, dict) and item.get("type") == "text"])
    #如果已经是字符串，直接返回
    return str(v) if v is not None else ""
#定义自适应字符串，通过前置校验器过滤非字符串内容
GradioString = Annotated[str, BeforeValidator(extract_gradio_text)]

class ChatRequest(BaseModel):
    query: GradioString
    session_id: str
#一个会话一个title
class SessionItem(BaseModel):
    session_id: str
    title: str | None = None