from typing import List
from langchain_core.embeddings import Embeddings
from openai import OpenAI

class CustomQwenEmbeddings(Embeddings):   #自定义Embedding类，彻底剥离LangChain针对OpenAI写的tiktoken校验和并发私货
    def __init__(   #初始化自定义客户端
        self, 
        api_key: str, 
        base_url: str, 
        model: str = "text-embedding-v3",   #向量模型名称
        ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    def embed_query(self, text: str) -> List[float]:   #处理单条文本
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding
    def embed_documents(self, texts: List[str]) -> List[List[float]]:    #处理文本列表
        embeddings = []   #稳妥的防爆策略：选择一条一条发
        for text in texts:   #LangChain自带的OpenAIEmbeddings会默认打包几百条一起发，极其容易触发国内API的并发拦截或长度超限
            embeddings.append(self.embed_query(text))   #实际业务中，这里甚至可以加上try...except和time.sleep()来应对API限流
        return embeddings