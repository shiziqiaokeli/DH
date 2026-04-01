#导入gradio库
import gradio as gr
#使用httpx进行异步流式请求
import httpx
#生成随机字符串作为session_id
import uuid

#当用户点击发送时
def user_submit(user_message, history):
    # 将 history 视为字典列表
    if history is None:
        history = []
    # 添加用户消息
    history.append({"role": "user", "content": user_message})
    # 添加一个空的助手占位符（新版中助手内容不能为空，可以给个空字符串）
    history.append({"role": "assistant", "content": ""})
    return "", history
#文本-文本聊天函数函数
async def text_text_chat(history, session_state):
    #获取刚刚user_submit追加的最后一条消息中的用户输入
    user_message = history[-2]["content"]
    #如果历史记录只有刚刚添加的这一条，说明是新对话
    if len(history) <= 2:
        #生成一个全新的session_id
        session_state["session_id"] = str(uuid.uuid4())
    #获取当前的session_id
    current_id = session_state["session_id"]
    #后端地址
    url = "http://127.0.0.1:8000/chat"
    #定义请求体
    payload = {
        "query": user_message,
        "session_id": current_id
    }
    #网络请求与容错处理
    try:
        #创建一个进入时自动打开退出时自动关闭且服务器60s未响应就自动关闭的异步客户端
        async with httpx.AsyncClient(timeout=60.0) as client:
            #发送POST请求（参数为url、请求体自动转译为json格式），并流式获取响应
            async with client.stream("POST", url, json=payload) as response:
                #防御性编程：检查后端是否报了 500 或 404 等非正常状态码
                if response.status_code != 200:
                    history[-1]["content"] = f"⚠️ 抱歉，后端大脑开小差了 (状态码: {response.status_code})"
                    yield history
                    return
                #遍历响应中的文本块
                async for chunk in response.aiter_text():
                    #获取当前内容
                    current_content = history[-1]["content"]
                    #无论content是字符串还是列表，都统一转为纯字符串处理
                    #解决了流式输出时，每行只有一个字的问题，根本原因在于Python语音的list+=str特性
                    #对列表的静默拆解把[{"text":"你好","type":"text"}]变成["你","好"]，导致每行只有一个字
                    #应该字符串合并然后一个字一个字输出，而不是拆解成一个个list拼接，再一个个渲染
                    if isinstance(current_content, list):
                        if len(current_content) > 0 and isinstance(current_content[0], dict):
                            text_value = current_content[0].get("text", "") + chunk
                        else:
                            text_value = chunk
                    else:
                        text_value = str(current_content) + chunk
                    history[-1]["content"] = [{"text": text_value, "type": "text"}]
                    yield history
    #捕获后端未启动的错误
    except httpx.ConnectError:
        history[-1]["content"] = "🔌 呼叫失败：无法连接到大脑服务器，请检查 FastAPI (端口 8000) 是否已启动。"
        yield history
    #捕获大模型思考太久的超时错误
    except httpx.TimeoutException:
        history[-1]["content"] = "⏳ 思考超时：问题太复杂了，或者网络延迟过高，请稍后再试。"
        yield history
    #捕获其他未知异常
    except Exception as e:
        history[-1]["content"] = f"❌ 系统异常: {str(e)}"
        yield history
#前端UI界面
custom_css = """
/*强制容器占满全宽，并去除内边距*/
.gradio-container { 
    max-width: 100% !important; 
    margin: 0 !important; 
    padding: 0 !important; 
}
/*移除内部主布局的默认间距*/
.main {
    padding: 0 !important;
}
/*左侧框架：浅灰色背景，无边框*/
#leftside {
    background-color: #f0f4f9; 
    padding: 20px 15px;
    height: 100vh;
}
/*右侧框架：设为相对定位，方便内部绝对定位*/
#rightside {
    position: relative;
    height: 100vh;
    display: flex;
    flex-direction: column;
}
/*顶部悬浮导航*/
#top-nav {
    position: absolute;
    top: 15px;
    left: 0;
    width: 100%;             /*必须撑满宽度，space-between 才会生效*/
    padding: 0 20px;         /*左右留出安全边距*/
    display: flex;
    justify-content: space-between; /*关键：分散对齐*/
    align-items: center;     /*垂直居中*/
    z-index: 1000;
    box-sizing: border-box;  /*确保 padding 不会撑破宽度*/
}
/*统一子项宽度，确保中间内容真正处于物理中心*/
.nav-item {
    flex: 1; 
}
.nav-center {
    text-align: center;
}
.nav-right {
    display: flex;
    justify-content: flex-end; /*将按钮推向最右侧*/
}
#mini-btn {
    /*取消自动撑满*/
    flex: 0 0 auto !important; 
    width: fit-content !important; 
    min-width: 0 !important;
    /*压缩内边距：让边框紧贴文字*/
    padding: 2px 10px !important; 
    /*如果是在Row/Column内部，防止它被拉伸*/
    align-self: center; 
}
/*居中且变窄的限制容器*/
#center-container {
    max-width: 850px; /*控制聊天和输入框的最大宽度*/
    margin: 0 auto;   /*水平居中*/
    height: 100vh;
    display: block;
    position: relative; /*为悬浮输入框提供参照*/
}
/*聊天框占满高度，沉入底层*/
#chat-window {
    height: 100% !important;
    padding-top: 80px !important; 
    /*底部留出大片空白！这样哪怕输入框挡在上面，用户也能把最后一条消息滚动出来*/
    padding-bottom: 200px !important; 
    border: none !important;
    background: transparent !important;
}
/*绝对定位的悬浮输入区域*/
#input-wrapper {
    position: absolute;
    bottom: 25px; /*距离底部25px固定*/
    left: 0;
    width: 100%;
    z-index: 999; /*浮在聊天记录之上*/
}
/*类似Gemini的输入*/
#input-card {
    background-color: #f0f4f9;
    border-radius: 24px;
    padding: 10px 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}
/*more下拉锚点：只占按钮宽度，供子菜单absolute参照*/
#more-menu-anchor {
    position: relative !important;
    width: fit-content !important;
    flex: 0 0 auto !important;
    align-self: center;
    min-width: 0 !important;
}
/*悬浮在按钮下方，不撑开顶栏Row*/
#more-menu-dropdown {
    position: absolute !important;
    top: 100% !important;
    right: 0 !important;
    margin-top: 6px !important;
    z-index: 1001 !important;
    flex-direction: column !important;
    gap: 8px !important;
    width: max-content !important;
    min-width: 140px;
    padding: 8px !important;
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
}
"""
#自带的clear不能直接物理删除Redis缓存，只是删除前端显示的聊天记录，同时会刷新页面，导致id变化，所以记忆会消失
with gr.Blocks(title="AI应用中台",fill_width=True) as demo:
    #用于记录当前对话的session_id
    session_state = gr.State(value=lambda: {"session_id": str(uuid.uuid4())})
    more_menu_open = gr.State(value=False)
    with gr.Row():#分左右
        with gr.Column(scale=1, elem_id="leftside") as sidebar_col:#（左侧）分上下
            with gr.Column(scale=1):#左上设计
                new_chat_btn = gr.Button("📝发起新对话", variant="primary", size="lg")
                gr.Markdown("""<div style="font-size: 16px; font-weight: 600; padding: 0;text-align: center;">
                对话
                </div>""")
                #创建其他聊天窗口按钮
            with gr.Column(scale=0):#左下设计
                rag_model_btn=gr.Button("知识库和模型", variant="secondary", size="lg")
        with gr.Column(scale=5, elem_id="rightside"):#（右侧）分上下
            with gr.Row(elem_id="top-nav"):#右上设计
                gr.Markdown("""<div class="nav-item" style="font-size: 16px; font-weight: 600; padding: 0;text-align: left;">
                string1
                </div>""")
                gr.Markdown("""<div class="nav-item nav-center" style="font-size: 16px; font-weight: 600; padding: 0;text-align: center;">
                string2
                </div>""")
                with gr.Row(elem_classes="nav-item nav-right"): 
                    with gr.Column(elem_id="more-menu-anchor"):
                        more_btn = gr.Button(
                            "more", variant="secondary", size="lg", elem_id="mini-btn"
                        )
                        with gr.Column(
                            elem_id="more-menu-dropdown", visible=False
                        ) as more_dropdown:
                            more_opt_a = gr.Button(
                                "重命名", variant="secondary", size="lg", elem_id="mini-btn"
                            )
                            more_opt_b = gr.Button(
                                "删除", variant="secondary", size="lg", elem_id="mini-btn"
                            )
            with gr.Column(elem_id="center-container"):#右下设计
                #聊天显示区
                chatbot = gr.Chatbot(
                    elem_id="chat-window",
                    show_label=False,
                    placeholder="<strong>string3</strong><br>string4",
                    editable=False,
                    buttons=[]
                )
                # 输入区和工具栏(顶层绝对定位，向上扩展)
                with gr.Column(elem_id="input-wrapper"):
                    with gr.Column(elem_id="input-card"):#输入框和工具栏上下对齐
                        msg_input = gr.Textbox(
                                placeholder="string5",
                                container=False, 
                                scale=9,
                                show_label=False,
                                lines=1,        
                                max_lines=6     # 向上最多扩展 8 行，底部不动
                            )   
                        #工具栏左右对齐
                        with gr.Row():
                            change_rag_btn = gr.Button("change_rag", size="lg",variant="secondary",elem_id="mini-btn")
                            change_prompt_btn = gr.Button("change_prompt", size="lg",variant="secondary",elem_id="mini-btn")
                            change_model_btn = gr.Button("change_model", size="lg",variant="secondary",elem_id="mini-btn")
                            with gr.Row(elem_classes="nav-item nav-right"): 
                                switch_btn = gr.Button("switch", size="lg",variant="secondary",elem_id="mini-btn")
                            submit_btn = gr.Button("🚀", size="lg",variant="primary",elem_id="mini-btn")               
    msg_input.submit(
        fn=user_submit,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
        queue=False  
    ).then(
        fn=text_text_chat,
        inputs=[chatbot, session_state],
        outputs=[chatbot] 
    )
    submit_btn.click(
        fn=user_submit,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
        queue=False
    ).then(
        fn=text_text_chat,
        inputs=[chatbot, session_state],
        outputs=[chatbot]
    )

    def toggle_more_menu(is_open: bool):
        new_open = not is_open
        return new_open, gr.update(visible=new_open)

    more_btn.click(
        fn=toggle_more_menu,
        inputs=[more_menu_open],
        outputs=[more_menu_open, more_dropdown],
        queue=False,
    )

    def reset_chat():
        return [], {"session_id": str(uuid.uuid4())}
    new_chat_btn.click(fn=reset_chat, inputs=[], outputs=[chatbot, session_state], queue=False)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        theme=gr.themes.Soft(primary_hue="blue"), 
        footer_links=[],#不显示FastAPI版权信息
        css=custom_css
    )