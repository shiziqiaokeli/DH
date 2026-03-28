#用于获得传入私有知识库资料的路径
from pathlib import Path
#通过路径和编码方式初始化加载器，调用加载器的load方法获取资料
from langchain_community.document_loaders import TextLoader
#通过块大小，重叠大小，分隔符初始化分割器，调用分割器的split_documents方法分割资料
from langchain_text_splitters import RecursiveCharacterTextSplitter
#导入自定义文本向量化类（彻底剥离LangChain针对OpenAI写的tiktoken校验和并发私货，适用于国内大模型）
from app.core.custom_embed import CustomQwenEmbeddings
#导入配置文件
from app.core.config import settings
#文本向量化后存入向量数据库
import os
from langchain_chroma import Chroma
#初始化大模型
from langchain_openai import ChatOpenAI
#导入提示词拼接工具和记忆区模块
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#构建有记忆的检索器（水龙头）
from langchain_classic.chains import create_history_aware_retriever
#接受前面所有chain的输出作为context（排水口）
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
#规定两个chain的协议（水管）
from langchain_classic.chains import create_retrieval_chain
#添加langchain标准的数据接口
from langchain_core.chat_history import BaseChatMessageHistory
#用于创建会话历史
from langchain_community.chat_message_histories import ChatMessageHistory
#将rag_chain包装成可运行的chain，并且自动添加会话历史
from langchain_core.runnables import RunnableWithMessageHistory

#用于获得传入私有知识库资料的路径
directory=Path(__file__).resolve().parent.parent.parent
path=directory / "data" / "njust_info.txt"
#通过路径和编码方式初始化加载器，调用加载器的load方法获取资料
doc_loader=TextLoader(path,encoding="utf-8")
documents=doc_loader.load()
#通过块大小，重叠大小，分隔符初始化分割器，调用分割器的split_documents方法分割资料
doc_splitter=RecursiveCharacterTextSplitter(
    chunk_size=300, 
    chunk_overlap=30,
    separators=["\n\n","\n","。","！","？","；","；","，",","," ",""]#中文特化分隔符
    )
chunks=doc_splitter.split_documents(documents)
#初始化文本向量化工具
embeddings=CustomQwenEmbeddings(
    api_key=settings.LLM_API_KEY,
    base_url=settings.LLM_BASE_URL,
    model="text-embedding-v1"
    )
#获取向量数据库路径
db_path = "./chroma_db"
#如果向量数据库存在且不为空，则加载向量数据库
if os.path.exists(db_path) and os.listdir(db_path):
    db = Chroma(
        embedding_function=embeddings,
        persist_directory=db_path
    )
#如果向量数据库不存在或者为空，则创建向量数据库
else:
    db = Chroma.from_documents(
        documents=chunks,
        embedding_function=embeddings,
        persist_directory=db_path
    )
#通过Top-K检索参数k初始化检索器
retriever=db.as_retriever(search_kwargs={"k": 3})
#初始化大模型
llm=ChatOpenAI(
    api_key=settings.LLM_API_KEY,
    base_url=settings.LLM_BASE_URL,
    model="qwen3.5-flash",
    temperature=0.1,#温度参数用于平衡模型的创造力和确定性
    streaming=True#控制模型流式传输
    )
#构建转化问题所用的系统提示词
transform_system_prompt=(
    "给定聊天历史和最新用户问题，"
    "该用户问题可能引用了聊天历史中的上下文（比如代词'它'）。"
    "请构建一个可以脱离聊天历史独立理解的全新问题。"
    "你只需要重写问题，如果不需要重写就原样返回，绝对不要尝试回答它。"
)
#把系统提示词，记忆区，用户问题拼接起来
transform_prompt = ChatPromptTemplate.from_messages([
    ("system", transform_system_prompt),#系统提示词
    MessagesPlaceholder("chat_history"),#记忆区，0轮时为空，每一轮对话后会自动添加到记忆区
    ("human", "{input}"),#用户问题
])
#构建有记忆的检索器（水龙头）
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, transform_prompt
)#把transform_prompt喂给llm得到转化后的问题，把转化后的问题通过retriever先向量化再检索出相似度最高的k个资料传给下一个chain
#构建回答问题所用的系统提示词
qa_system_prompt="""你是一名知识渊博的南京理工大学（NJUST）资深校友。请根据提供的【校史参考资料】来回答【校友提问】。
要求：
1. 仅根据资料内容回答，不要胡编乱造。
2. 如果资料中没提到相关信息，请礼貌地回答：“抱歉，这段校史我还需要再查证一下。”
3. 回答语气要亲切、严谨，体现出南理工“献身”精神的底蕴。
【校史参考资料】：{context}"""
#把系统提示词，记忆区，用户问题拼接起来
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
#接受前面所有chain的输出作为context（排水口），赋值给qa_prompt，然后调用llm回答问题
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
#规定两个chain的协议（水管）
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
#全局字典，用于存储会话历史
store={}
#如果没有会话id，则创建一个会话历史，否则返回对应的会话历史
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
#将rag_chain包装成可运行的chain，并且自动添加会话历史
final_chain=RunnableWithMessageHistory(
    rag_chain,
    get_session_history,#用于管理会话历史
    input_messages_key="input",          
    history_messages_key="chat_history", 
    output_messages_key="answer",       
)
'''
async def main():
    queries = [
        "南京理工大学的首任院长是谁？",
        "他是什么军衔？"
    ]
    for query in queries:
        print(f"校友提问: {query}\n")
        print("资深校友正在回忆中...")
        # 注意：现在传给 astream 的input必须是一个字典 {"input": query}
        # 同时config也是一个字典{"configurable": {"session_id": "admin"}}
        async for chunk in final_chain.astream({"input": query},config={"configurable": {"session_id": "admin"}}):
            if "answer" in chunk:
                print(chunk["answer"], end="", flush=True)
        print("\n\n---回答结束---")
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
'''
class RAGService:
    def __init__(self):
        # 将你的初始化逻辑（embeddings, db, chain）放进来
        self.chain = final_chain 

    async def chat(self, query: str, session_id: str):
        async for chunk in self.chain.astream(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        ):
            if "answer" in chunk:
                yield chunk["answer"]