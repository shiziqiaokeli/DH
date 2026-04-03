from pathlib import Path
import gradio as gr
from screen_handlers import (
    apply_rename,
    close_rename_dialog,
    delete_current_chat,
    on_page_load,
    on_session_pick,
    open_rename_dialog,
    refresh_session_radio_after_reply,
    reset_chat,
    text_text_chat,
    toggle_more_menu,
    user_submit,
)
_CSS_PATH = Path(__file__).resolve().parent / "styles" / "screen.css"
SCREEN_CSS = _CSS_PATH.read_text(encoding="utf-8")
#前端UI界面
#自带的clear不能直接物理删除Redis缓存，只是删除前端显示的聊天记录，同时会刷新页面，导致id变化，所以记忆会消失
with gr.Blocks(title="AI应用中台",fill_width=True) as demo:
    #用于记录当前对话的session_id
    browser_id = gr.BrowserState(
        default_value={"session_id": None},
        storage_key="dh_chat_session",
        secret="bondageoflife",
        )
    session_state = gr.State(value={"session_id": None})
    more_menu_open = gr.State(value=False)
    display_names = gr.State(value={})  # session_id -> display_name 的映射
    with gr.Row():#分左右
        with gr.Column(scale=1, elem_id="leftside") as sidebar_col:#（左侧）分上下
            with gr.Column(scale=1):#左上设计
                new_chat_btn = gr.Button("📝发起新对话", variant="primary", size="lg")
                new_rag_btn = gr.UploadButton(
                    "新建知识库",  
                    variant="secondary",
                    size="lg",
                    file_count="single", 
                    file_types=[".txt",] 
                )
                new_prompt_btn = gr.Button("新建提示词", variant="secondary", size="lg")
                new_model_btn = gr.Button("训练模型", variant="secondary", size="lg")
                gr.Markdown("""<div style="font-size: 16px; font-weight: 600; padding: 0;text-align: center;">
                对话
                </div>""")
                session_radio = gr.Radio(
                    label="",
                    choices=[],
                    value=None,
                    show_label=False
                    )
                
            with gr.Column(scale=0):#左下设计
                rag_model_btn=gr.Button(" ⚙️ 设 置 ", variant="secondary", size="lg")
        with gr.Column(scale=5, elem_id="rightside"):#（右侧）分上下
            with gr.Row(elem_id="top-nav"):#右上设计
                gr.Markdown("""<div class="nav-item" style="font-size: 16px; font-weight: 600; padding: 0;text-align: left;">
                人工智能机器人
                </div>""")
                title_centre=gr.Markdown("""<div class="nav-item nav-center" style="font-size: 16px; font-weight: 600; padding: 0;text-align: center;">
                default
                </div>""")
                with gr.Row(elem_classes="nav-item nav-right"): 
                    with gr.Column(elem_id="more-menu-anchor"):
                        more_btn = gr.Button(
                            "更多", variant="secondary", size="lg", elem_id="mini-btn"
                        )
                        with gr.Column(
                            elem_id="more-menu-dropdown", visible=False
                        ) as more_dropdown:
                            rename_btn = gr.Button(
                                "重命名", variant="secondary", size="lg", elem_id="mini-btn"
                            )
                            delete_btn = gr.Button(
                                "删除", variant="secondary", size="lg", elem_id="mini-btn"
                            )
                            # 重命名弹窗（手写遮罩 + 卡片）
                            with gr.Column(visible=False, elem_id="rename-overlay") as rename_overlay:
                                with gr.Column(elem_id="rename-card"):
                                    gr.Markdown('<div id="rename-title">重命名会话</div>')
                                    rename_input = gr.Textbox(
                                        label="",
                                        show_label=False,
                                        placeholder="输入新名称",
                                        elem_id="rename-input"
                                    )
                                    with gr.Row(elem_id="rename-buttons"):
                                        confirm_rename_btn = gr.Button("确认", variant="primary", elem_classes="rename-btn")
                                        cancel_rename_btn = gr.Button("取消", variant="secondary", elem_classes="rename-btn")
            with gr.Column(elem_id="center-container"):#右下设计
                #聊天显示区
                chatbot = gr.Chatbot(
                    elem_id="chat-window",
                    show_label=False,
                    placeholder='<span style="font-size: 24px;">你好</span><br><span style="font-size: 30px;"><strong>需要我为你做些什么？</strong></span>',
                    editable=False,
                    buttons=[]
                )
                # 输入区和工具栏(顶层绝对定位，向上扩展)
                with gr.Column(elem_id="input-wrapper"):
                    with gr.Column(elem_id="input-card"):#输入框和工具栏上下对齐
                        msg_input = gr.Textbox(
                                placeholder="你想聊些什么",
                                container=False, 
                                scale=9,
                                show_label=False,
                                lines=1,        
                                max_lines=6     # 向上最多扩展 8 行，底部不动
                            )   
                        #工具栏左右对齐
                        with gr.Row():
                            change_rag_btn = gr.Button("rag", size="lg",variant="secondary",elem_id="mini-btn")
                            change_prompt_btn = gr.Button("prompt", size="lg",variant="secondary",elem_id="mini-btn")
                            change_temperature_btn = gr.Button("t", size="lg",variant="secondary",elem_id="mini-btn")
                            change_model_btn = gr.Button("model", size="lg",variant="secondary",elem_id="mini-btn")
                            with gr.Row(elem_classes="nav-item nav-right"): 
                                switch_btn = gr.Button("文本/语音输出", size="lg",variant="secondary",elem_id="mini-btn") 
                            audio_submit_btn = gr.Button("🎙️", size="lg",variant="primary",elem_id="mini-btn") 
                            submit_btn = gr.Button("🛩️", size="lg",variant="primary",elem_id="mini-btn") 
    
    demo.load(
        on_page_load,
        inputs=[browser_id, session_state],
        outputs=[chatbot, session_state, browser_id, session_radio, title_centre, display_names]
    )        
    session_radio.change(
        fn=on_session_pick,
        inputs=[session_radio, session_state, browser_id],
        outputs=[chatbot, session_state, browser_id, title_centre, display_names],
        queue=False,
    )
    def wire_send_message():
        for trigger in (msg_input.submit, submit_btn.click):
            trigger(
                fn=user_submit,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot],
                queue=False,
            ).then(
                fn=text_text_chat,
                inputs=[chatbot, session_state],
                outputs=[chatbot],
            ).then(
                fn=refresh_session_radio_after_reply,
                inputs=[session_state, display_names],
                outputs=[session_radio, title_centre, display_names],
            )
    wire_send_message()
    
    more_btn.click(
        fn=toggle_more_menu,
        inputs=[more_menu_open],
        outputs=[more_menu_open, more_dropdown],
        queue=False,
    )
    delete_btn.click(
        fn=delete_current_chat,
        inputs=[session_state, browser_id, display_names],
        outputs=[chatbot, session_radio, session_state, browser_id, title_centre, display_names],
        queue=False,
    )
    new_chat_btn.click(
        fn=reset_chat,
        inputs=[browser_id, display_names],
        outputs=[chatbot, session_state, browser_id, session_radio, title_centre, display_names],
        queue=False,
    )
    # 打开重命名弹窗
    rename_btn.click(
        fn=open_rename_dialog,
        inputs=[session_state, display_names],
        outputs=[rename_input, rename_overlay],
        queue=False,
        show_progress="hidden",
    )
    # 取消重命名
    cancel_rename_btn.click(
        fn=close_rename_dialog,
        inputs=None,
        outputs=[rename_input, rename_overlay],
        queue=False,
        show_progress="hidden",
        js="""() => { setTimeout(() => { window.location.reload(); }, 100); }"""
    )
    # 确认重命名
    confirm_rename_btn.click(
        fn=apply_rename,
        inputs=[rename_input, session_state, display_names],
        outputs=[display_names, title_centre, session_radio, rename_input, rename_overlay],
        queue=False,
    ).then(
        # 刷新 Radio 列表以更新显示
        fn=refresh_session_radio_after_reply,
        inputs=[session_state, display_names],
        outputs=[session_radio, title_centre, display_names],
        show_progress="hidden",
        js="""() => { setTimeout(() => { window.location.reload(); }, 100); }"""
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        theme=gr.themes.Soft(primary_hue="blue"), 
        footer_links=[],#不显示FastAPI版权信息
        css=SCREEN_CSS
    )