from typing import Annotated, Any, List, Dict
from pydantic import BaseModel, BeforeValidator

def extract_gradio_text(v: Any) -> str:
    # 只处理 Gradio 6.0 的文本列表格式: [{'text': '...', 'type': 'text'}]
    if isinstance(v, list):
        # 提取列表中所有 type 为 text 的内容并拼接
        return "".join([str(item.get("text", "")) for item in v if isinstance(item, dict) and item.get("type") == "text"])
    
    # 如果已经是字符串（比如旧版前端发来的），直接返回
    return str(v) if v is not None else ""

# 定义自适应字符串
GradioString = Annotated[str, BeforeValidator(extract_gradio_text)]

class ChatRequest(BaseModel):
    query: GradioString  # 进来的数据如果是列表，会被自动洗成 str
    session_id: str