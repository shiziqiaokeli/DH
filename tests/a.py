import inspect
import gradio as gr
#print(inspect.signature(gr.BrowserState))
with gr.Modal(visible=False) as rename_modal:
    rename_input = gr.Textbox(
            label="会话显示名称",
            show_label=True,
            lines=1,
        )
    with gr.Row():
            rename_confirm_btn = gr.Button("确认", variant="primary")
            rename_cancel_btn = gr.Button("取消", variant="secondary")
        
        