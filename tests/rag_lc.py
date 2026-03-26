import string
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from app.core.config import settings
from openai import AsyncOpenAI
from app.core.custom_embed import CustomQwenEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import asyncio

base_dir = Path(__file__).resolve().parent.parent
file_path = base_dir / "data" / "njust_info.txt"

documents = TextLoader(file_path,encoding="utf-8").load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = text_splitter.split_documents(documents)

embeddings = CustomQwenEmbeddings(
    api_key=settings.LLM_API_KEY,
    base_url=settings.LLM_BASE_URL, 
    model="text-embedding-v1"
)

db=Chroma.from_documents(chunks, embeddings, persist_directory="./langchain_chroma_db")

llm=ChatOpenAI(
    api_key=settings.LLM_API_KEY,
    base_url=settings.LLM_BASE_URL,
    model="qwen3.5-flash",
    temperature=0.1,
    streaming=True)

retriever=db.as_retriever(search_kwargs={"k": 3})

string="""
你是一名知识渊博的南京理工大学（NJUST）资深校友。
请根据提供的【校史参考资料】来回答【校友提问】。

要求：
1. 仅根据资料内容回答，不要胡编乱造。
2. 如果资料中没提到相关信息，请礼貌地回答：“抱歉，这段校史我还需要再查证一下。”
3. 回答语气要亲切、严谨，体现出南理工“献身”精神的底蕴。

【校史参考资料】：
{context}

【校友提问】：
{question}

资深校友的回答：
"""

prompt=ChatPromptTemplate.from_template(string)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# 1. {"context": retriever, "question": RunnablePassthrough()}：
#    - 构建一个映射，把输入的问题(query)传递给 "question"，同时用同样的输入(query)查询知识库(retriever)获得相关上下文，作为 "context"。
#    - retriever 返回与问题相关的文档片段，"RunnablePassthrough" 让原始问题原样传递。
# 2. |prompt：将上一步得到的 {context, question} 传递给 prompt，进行标准化的提示模版填充。
# 3. |llm：将填充后的 prompt 输入到大模型（llm），让其生成答案（以流式输出）。
# 4. |StrOutputParser()：对大模型输出的内容做字符串格式化处理，方便后续处理或显示。
# 连起来，即：输入一个问题，自动查找相关知识片段与问题一同填充到对话模板，传给 LLM 生成流式答案，输出最终字符串。    
async def main():
    query = "中国人民解放军炮兵工程学院在哪年成立？"
    print(f"校友提问: {query}\n")
    print("资深校友正在回忆中...")
    # 使用 async for 处理 astream 结果
    async for chunk in chain.astream(query):
        print(chunk, end="", flush=True)
    print("\n\n---回答结束---")

if __name__ == "__main__":
    asyncio.run(main())
   