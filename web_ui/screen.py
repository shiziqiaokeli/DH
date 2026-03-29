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
    #创建一个进入时自动打开退出时自动关闭且服务器60s未响应就自动关闭的异步客户端
    async with httpx.AsyncClient(timeout=60.0) as client:
        #发送POST请求（参数为url、请求体自动转译为json格式），并流式获取响应
        async with client.stream("POST", url, json=payload) as response:
            #遍历响应中的文本块
            async for chunk in response.aiter_text():
                #将部分消息拼接到partial_message中
                partial_message += chunk
                #视觉上流式输出，实际上迭代一个不停变长的字符串
                yield partial_message
#搭建一个demo容器
with gr.Blocks(title="南理工校友助手") as demo:
    #表示二级标题（中间要有空格）
    gr.Markdown("## 南京理工大学校史智能助手")
    session_state = gr.State(value=lambda: {"session_id": str(uuid.uuid4())})
    #全自动聊天机器人
    #输入框
    #展示区：显示历史聊天记录
    #发送按钮
    #逻辑：点击发送按钮后，把输入框的内容传给fn的第一个参数，同时把聊天历史传给fn的第二个参数，最后把session_state传给fn的第三个参数
    gr.ChatInterface(
        fn=text_text_chat,
        additional_inputs=[session_state]
        #自带的clear不能直接物理删除Redis缓存，只是删除前端显示的聊天记录，同时会刷新页面，导致id变化，所以记忆会消失
        )

if __name__ == "__main__":
    demo.launch(server_port=7860)