#导入gradio库
import gradio as gr
#使用httpx进行异步流式请求
import httpx
#生成随机字符串作为session_id
import uuid

#文本-文本聊天函数函数
async def text_text_chat(message, history, session_state):
    #刚打开页面或者新开一个对话时
    if len(history) == 0:
        #生成一个全新的session_id
        session_state["session_id"] = str(uuid.uuid4())
    #获取当前的session_id
    current_id = session_state["session_id"]
    #后端地址
    url = "http://127.0.0.1:8000/chat"
    #定义请求体
    payload = {
        "query": message,
        "session_id": current_id
    }
    #定义一个空字符串用于存储部分消息
    partial_message = ""
    #网络请求与容错处理
    try:
        #创建一个进入时自动打开退出时自动关闭且服务器60s未响应就自动关闭的异步客户端
        async with httpx.AsyncClient(timeout=60.0) as client:
            #发送POST请求（参数为url、请求体自动转译为json格式），并流式获取响应
            async with client.stream("POST", url, json=payload) as response:
                #防御性编程：检查后端是否报了 500 或 404 等非正常状态码
                if response.status_code != 200:
                    yield f"⚠️ 抱歉，后端大脑开小差了 (状态码: {response.status_code})"
                    return
                #遍历响应中的文本块
                async for chunk in response.aiter_text():
                    #将部分消息拼接到partial_message中
                    partial_message += chunk
                    #视觉上流式输出，实际上迭代一个不停变长的字符串
                    yield partial_message
    #捕获后端未启动的错误
    except httpx.ConnectError:
        yield "🔌 呼叫失败：无法连接到大脑服务器，请检查 FastAPI (端口 8000) 是否已启动。"
    #捕获大模型思考太久的超时错误
    except httpx.TimeoutException:
        yield "⏳ 思考超时：问题太复杂了，或者网络延迟过高，请稍后再试。"
    #捕获其他未知异常
    except Exception as e:
        yield f"❌ 系统异常: {str(e)}"
#前端UI区
with gr.Blocks(title="南理工校友助手") as demo:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h2>🎓 南京理工大学校史智能助手</h2>
            <p>基于 LangChain RAG 技术驱动，解答您关于南理工的各种问题</p>
        </div>
        """
    )
    #状态保持：使用 lambda 确保每个新打开网页的用户都有独立的初始字典
    session_state = gr.State(value=lambda: {"session_id": str(uuid.uuid4())})
    #全自动聊天机器人
    #输入框
    #展示区：显示历史聊天记录
    #发送按钮
    #逻辑：点击发送按钮后，把输入框的内容传给fn的第一个参数，同时把聊天历史传给fn的第二个参数，最后把session_state传给fn的第三个参数
    gr.ChatInterface(
        fn=text_text_chat,
        additional_inputs=[session_state],
        #自带的clear不能直接物理删除Redis缓存，只是删除前端显示的聊天记录，同时会刷新页面，导致id变化，所以记忆会消失
        chatbot=gr.Chatbot(height=550, placeholder="<strong>提问示例：</strong><br>1. 信息自动化学院是哪年成立的？<br>2. 后来它改名了吗？"),
        textbox=gr.Textbox(placeholder="请输入您的问题...", container=False, scale=7),
        )
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())