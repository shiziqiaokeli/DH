from app.core.config import settings  # 导入全局配置对象
import os
import asyncio                      # 导入异步支持模块
from openai import AsyncOpenAI      # 导入 OpenAI 异步客户端

# 初始化异步 OpenAI 客户端，配置 API Key 与 Base URL
client = AsyncOpenAI(
    api_key=settings.LLM_API_KEY,
    base_url=settings.LLM_BASE_URL,
)

async def multi_turn_chat():
    """
    多轮对话演示函数，模拟用户与 AI 助手的连续交流
    """
    print(f"\n========== 多轮对话 ==========")

    # 会话历史，系统提示词用于规定 AI 行为
    history = [
        {"role": "system", "content": "你是一个AI助手。"}
    ]

    # 第一轮对话：用户提出自己的喜欢
    input_1 = "你好，我最喜欢的水果是苹果。"
    print(f"用户: {input_1}")
    history.append({"role": "user", "content": input_1})  # 用户输入加入历史

    # 向大模型发起第一次流式对话请求
    response_1 = await client.chat.completions.create(
        model=settings.LLM_MODEL_NAME,
        messages=history,
        stream=True,  # 打开流式输出，降低首包延迟
    )

    print("AI: ", end="", flush=True)
    full_content = ""  # 用于累积 AI 的完整回复内容

    # 异步遍历流式响应，将内容输出并拼接保存
    async for chunk in response_1:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            full_content += content
    print()  # 换行

    # 将 AI 的回复也加入对话历史，保证上下文连贯
    history.append({"role": "assistant", "content": full_content})

    # 第二轮对话：考察 AI 是否记得用户信息
    input_2 = "考考你的记忆力，我最喜欢的水果是什么？"
    print(f"\n用户: {input_2}")

    history.append({"role": "user", "content": input_2})  # 用户输入加入历史

    # 向大模型发起第二次流式对话请求
    response_2 = await client.chat.completions.create(
        model=settings.LLM_MODEL_NAME,
        messages=history,
        stream=True,
    )
    
    print("AI: ", end="", flush=True)

    # 输出 AI 回复（无需累计内容）
    async for chunk in response_2:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print()

if __name__ == "__main__":
    # 异步运行多轮对话测试
    asyncio.run(multi_turn_chat())