from typing import List
from langchain_core.embeddings import Embeddings
from openai import OpenAI

#自定义Embedding类，彻底剥离LangChain针对OpenAI写的tiktoken校验和并发私货。
#适用于通义千问、Kimi、智谱等所有兼容OpenAI格式的国内大模型API。
class CustomQwenEmbeddings(Embeddings):
    #初始化原生的OpenAI客户端
    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    #处理单条文本
    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    #处理文本列表
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        #稳妥的防爆策略：选择一条一条发。
        #LangChain自带的OpenAIEmbeddings会默认打包几百条一起发，极其容易触发国内API的并发拦截或长度超限。
        for text in texts:
            #实际业务中，这里甚至可以加上try...except和time.sleep()来应对API限流
            embeddings.append(self.embed_query(text))
        return embeddings