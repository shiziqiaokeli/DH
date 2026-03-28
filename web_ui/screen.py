import gradio as gr
import httpx
import asyncio

async def predict(message, history):
    # 后端地址
    url = "http://127.0.0.1:8000/chat"
    payload = {
        "query": message,
        "session_id": "gradio_user" # 实际开发可以动态生成
    }
    
    partial_message = ""
    # 使用 httpx 进行异步流式请求
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, json=payload) as response:
            async for chunk in response.aiter_text():
                partial_message += chunk
                yield partial_message

# 使用 Gradio 原生的 ChatInterface
with gr.Blocks(title="南理工校友助手") as demo:
    gr.Markdown("## 🛡️ 南京理工大学校史智能助手")
    gr.ChatInterface(
        fn=predict,
        examples=["华东工程学院哪年成立？", "那它后来改名了吗？"]
        # 如果你想自定义按钮文字，最新版通常在组件初始化后修改，
        # 初学者建议先删掉自定义按钮参数，跑通流程。
        )

if __name__ == "__main__":
    demo.launch(server_port=7860)