import uuid
import gradio as gr
import httpx

BASE = "http://127.0.0.1:8000"
TIMEOUT_CHAT = 60.0
TIMEOUT_HISTORY = 30.0
TIMEOUT_API = 15.0

def _label_from_session_item(it: dict) -> str:
    """title 供人读；未设置时用 session_id"""
    sid = it["session_id"]
    raw = (it.get("title") or "").strip()
    return raw if raw else sid
#提交函数
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
def centre_title_html(session_id: str | None, display_names: dict | None = None) -> str:
    """生成顶部标题 HTML，优先使用自定义显示名称"""
    display_names = display_names or {}
    if session_id:
        # 优先使用自定义名称，没有则显示 session_id
        label = display_names.get(session_id, session_id)
    else:
        label = "default"
    return (
        '<div class="nav-item nav-center" style="font-size: 16px; font-weight: 600; '
        'padding: 0;text-align: center;">'
        f"{label}"
        "</div>"
    )   
#拉取会话列表
async def fetch_session_choices(
    preferred_session_id: str | None = None,
    display_names: dict | None = None,
) -> tuple:
    display_names = display_names or {}
    title_none = gr.update(value=centre_title_html(None, display_names))
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(f"{BASE}/sessions")
            if r.status_code != 200:
                return gr.update(choices=[], value=None), title_none, {}

            items = r.json()
            session_ids = {it["session_id"] for it in items}
            merged: dict[str, str] = {}
            for it in items:
                sid = it["session_id"]
                merged[sid] = _label_from_session_item(it)
            for sid, lab in display_names.items():
                if sid not in merged:
                    merged[sid] = lab

            choices = [
                (merged.get(it["session_id"], _label_from_session_item(it)), it["session_id"])
                for it in items
            ]

            if preferred_session_id and preferred_session_id not in session_ids:
                lab = merged.get(preferred_session_id, preferred_session_id)
                choices.append((lab, preferred_session_id))

            choice_values = {c[1] for c in choices}
            if preferred_session_id and preferred_session_id in choice_values:
                val = preferred_session_id
            elif choices:
                val = choices[-1][1]
            else:
                val = None

            return (
                gr.update(choices=choices, value=val),
                gr.update(value=centre_title_html(val, merged)),
                merged,
            )
    except httpx.ConnectError:
        return gr.update(choices=[], value=None), title_none, {}
#加载上一次会话
async def on_page_load(browser_data, session_state): 
    data = dict(browser_data) if browser_data else {}
    cid = data.get("session_id")
    if not cid:
        cid = str(uuid.uuid4())
        data["session_id"] = cid
    session_state = dict(session_state) if session_state else {}
    session_state["session_id"] = cid
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{BASE}/history/{cid}")
            if r.status_code == 200:
                history = r.json()
            else:
                history = []
    except httpx.ConnectError:
        history = []

    radio_update, title_update, merged_names = await fetch_session_choices(
        preferred_session_id=cid, display_names={}
    )
    return history, session_state, data, radio_update, title_update, merged_names
#(物理)删除函数
async def delete_current_chat(session_state, browser_data, display_names: dict):
    sid = session_state.get("session_id") if session_state else None
    if sid:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                await client.post(
                    f"{BASE}/delete",
                    json={"query": "", "session_id": sid},
                )
        except httpx.ConnectError:
            pass
        # 删除显示名称映射
        display_names = dict(display_names) if display_names else {}
        display_names.pop(sid, None)
    
    new_sid = str(uuid.uuid4())
    new_state = {"session_id": new_sid}
    radio_update, title_update, merged_names = await fetch_session_choices(
        preferred_session_id=new_sid, display_names={}
    )
    data = dict(browser_data) if browser_data else {}
    data["session_id"] = new_sid
    return [], radio_update, new_state, data, title_update, merged_names
async def on_session_pick(session_id, session_state, browser_data):
    session_state = dict(session_state) if session_state else {}
    data = dict(browser_data) if browser_data else {}
    if not session_id:
        return [], session_state, data, gr.update(value=centre_title_html(None)), {}
    session_state["session_id"] = session_id
    data["session_id"] = session_id
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{BASE}/history/{session_id}")
            history = r.json() if r.status_code == 200 else []
    except httpx.ConnectError:
        history = []
    merged: dict[str, str] = {}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(f"{BASE}/sessions")
            if r.status_code == 200:
                for it in r.json():
                    merged[it["session_id"]] = _label_from_session_item(it)
    except httpx.ConnectError:
        pass
    return (
        history,
        session_state,
        data,
        gr.update(value=centre_title_html(session_id, merged)),
        merged,
    )

async def refresh_session_radio_after_reply(session_state, display_names: dict | None):
    sid = (session_state or {}).get("session_id")
    radio_u, title_u, merged = await fetch_session_choices(
        preferred_session_id=sid, display_names=display_names
    )
    return radio_u, title_u, merged

def open_rename_dialog(session_state, display_names):
    """打开重命名弹窗，输入框默认填入当前显示名称或 session_id"""
    sid = session_state.get("session_id") if session_state else None
    default_value = display_names.get(sid, sid) if sid else ""
    return gr.update(value=default_value), gr.update(visible=True)

def close_rename_dialog():
    """关闭重命名弹窗：仅隐藏遮罩；输入框保持可见，避免 Gradio 对 Textbox 误用 visible。"""
    return gr.update(), gr.update(visible=False)

async def apply_rename(new_name: str, session_state, display_names: dict):
    sid = session_state.get("session_id") if session_state else None
    if not sid or not new_name or not new_name.strip():
        return (
            display_names,
            gr.update(value=centre_title_html(sid, display_names)),
            gr.update(),
            gr.update(),
            gr.update(visible=False),
        )

    text = new_name.strip()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_API) as client:
            await client.put(
                f"{BASE}/sessions/{sid}/title",
                json={"session_id": sid, "title": text},
            )
    except httpx.ConnectError:
        pass

    display_names = dict(display_names) if display_names else {}
    display_names[sid] = text

    return (
        display_names,
        gr.update(value=centre_title_html(sid, display_names)),
        gr.update(),
        gr.update(),
        gr.update(visible=False),
    )
def toggle_more_menu(is_open: bool):
    new_open = not is_open
    return new_open, gr.update(visible=new_open)
async def reset_chat(browser_data, display_names: dict):
    data = dict(browser_data) if browser_data else {}
    cid = str(uuid.uuid4())
    data["session_id"] = cid
    new_state = {"session_id": cid}
    radio_update, title_update, merged_names = await fetch_session_choices(
        preferred_session_id=cid, display_names=display_names
    )
    return [], new_state, data, radio_update, title_update, merged_names

def on_kb_file_selected(file):
    """用户选完文件后：暂存文件引用，并显示名称框与确认按钮。"""
    if file is None:
        return (
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False),
        )
    return (
        file,
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(
            value="已选择文件，请填写名称后点击 **确认上传**。",
            visible=True,
        ),
    )

async def confirm_kb_upload(pending_file, kb_name: str) -> str:
    """第二步：带名称提交到后端（复用原有 upload_rag_file）。"""
    if pending_file is None:
        return "⚠️ 请先在上方选择文件"
    return await upload_rag_file(pending_file, kb_name)

async def upload_rag_file(file, kb_name: str) -> str:
    if not kb_name or not kb_name.strip():
        return "⚠️ 请先输入知识库名称"
    if file is None:
        return "⚠️ 请选择文件"
    async with httpx.AsyncClient(timeout=120.0) as client:
        with open(file.name, "rb") as f:
            response = await client.post(
                f"{BASE}/rag/upload",
                data={"kb_name": kb_name.strip()},
                files={"file": (file.name.split("\\")[-1], f, "text/plain")},
                timeout=None,
            )
    if response.status_code == 200:
        return f"✅ 知识库「{response.json().get('name')}」创建成功"
    return f"❌ 创建失败：{response.text}"

def on_prompt_form_open():
    """点击「新建提示词」后展开输入区。"""
    return (
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(value="请填写提示词后点击 **确认保存**。", visible=True),
    )


async def confirm_save_prompt(prompt_body: str, prompt_name: str) -> str:
    """把用户填的 prompt 正文 + 名称 POST 到后端写入 MySQL。"""
    if not prompt_body or not prompt_body.strip():
        return "⚠️ 请填写提示词正文"
    if not prompt_name or not prompt_name.strip():
        return "⚠️ 请填写提示词名称"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{BASE}/prompts",
                data={
                    "prompt_name": prompt_name.strip(),
                    "prompt_body": prompt_body.strip(),
                },
            )
        if response.status_code == 200:
            return f"✅ 提示词「{response.json().get('name')}」保存成功"
        return f"❌ 保存失败：{response.text}"
    except httpx.ConnectError:
        return "🔌 无法连接到后端服务器"
    except Exception as e:
        return f"❌ 异常：{str(e)}"


