from app.core.config import settings  # 导入全局配置对象
import os
import asyncio                      # 导入异步支持模块
from openai import AsyncOpenAI      # 导入 OpenAI 异步客户端

# 初始化异步 OpenAI 客户端，配置 API Key 与 Base URL
client = AsyncOpenAI(
    api_key=settings.LLM_API_KEY,
    base_url=settings.LLM_BASE_URL,
)

async def single_turn_chat(input: str):
    """
    单轮对话函数，向大模型发起 streaming 聊天请求并输出响应内容。

    参数:
        input (str): 用户输入
    返回值:
        None
    """

    print(f"\n========== 单轮对话 ==========")
    print(f"用户: {input}")
    print("AI: ", end="", flush=True)  # 保持 AI 输出在同一行

    try:
        # 调用 openai 聊天接口，使用 stream=True 实现流式输出，降低首包延迟
        response = await client.chat.completions.create(
            model="qwen3.5-flash",
            messages=[
                {"role": "system", "content": "你是一个说话幽默的资深后端程序员。"},
                {"role": "user", "content": input}
            ],
            temperature=0.7,
            stream=True,
        )

        # 异步遍历流式响应，每次只输出本次新增的文本内容
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)

    except Exception as e:
        # 捕获异常并提示错误，避免抛出不明确的异常
        print(f"Error: {e}")

if __name__ == "__main__":
    # 以异步方式运行单轮对话测试
    asyncio.run(single_turn_chat("给我讲一个关于 Python 的冷笑话。"))