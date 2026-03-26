from typing import List
from langchain_core.embeddings import Embeddings
from openai import OpenAI

class CustomQwenEmbeddings(Embeddings):
    """
    自定义 Embedding 类，彻底剥离 LangChain 针对 OpenAI 写的 tiktoken 校验和并发私货。
    适用于通义千问、Kimi、智谱等所有兼容 OpenAI 格式的国内大模型 API。
    """
    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-v1"):
        # 1. 初始化原生的 OpenAI 客户端，没有任何 LangChain 的中间商赚差价
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_query(self, text: str) -> List[float]:
        """
        核心方法 1：处理单条文本
        当用户在前端输入“信息自动化与制造工程学院在哪年成立？”时，LangChain 会自动调用这个方法。
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        核心方法 2：处理文本列表
        当我们把南理工校史切分成几百个 chunk，调用 Chroma.from_documents() 时，LangChain 会调用这个方法。
        """
        embeddings = []
        # 稳妥的防爆策略：我们选择一条一条发。
        # LangChain 自带的 OpenAIEmbeddings 会默认打包几百条一起发，极其容易触发国内 API 的并发拦截或长度超限。
        for text in texts:
            # 实际业务中，这里甚至可以加上 try...except 和 time.sleep() 来应对 API 限流
            embeddings.append(self.embed_query(text))
        return embeddings