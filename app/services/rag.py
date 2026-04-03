#用于获得传入私有知识库资料的路径
from pathlib import Path
import asyncio
import uuid
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
from app.core.config import settings
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
#将会话历史存入Redis
from langchain_community.chat_message_histories import RedisChatMessageHistory
#添加langchain标准的数据接口
from langchain_core.chat_history import BaseChatMessageHistory
#用于创建会话历史
from langchain_community.chat_message_histories import ChatMessageHistory
#将rag_chain包装成可运行的chain，并且自动添加会话历史
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
#获取向量数据库路径
db_path = settings.VECTOR_DB_PATH

_chain_cache: dict[str, RunnableWithMessageHistory] = {}

#初始化大模型
def _make_llm(temperature: float) -> ChatOpenAI:
    """按 system_settings.t_value 构建 LLM。"""
    return ChatOpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model="qwen3.5-flash",
        temperature=temperature,
        streaming=True,
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
])#把transform_prompt喂给llm得到转化后的问题，把转化后的问题通过retriever先向量化再检索出相似度最高的k个资料传给下一个chain
#获得Redis的URL
redis_url=settings.REDIS_URL
#根据session_id获取会话历史
def get_session_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url=redis_url
    )

async def process_uploaded_file(file_path: str) -> str:
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=30,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ",", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    collection_name = f"kb_{uuid.uuid4().hex[:12]}"   # 后端自动生成
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=db_path,
    )
    await asyncio.to_thread(db.add_documents, chunks)
    return collection_name   # 返回给调用方，让它存进 MySQL

def _build_chain(collection_name: str, qa_system_prompt: str, temperature: float) -> RunnableWithMessageHistory:
    llm = _make_llm(temperature)
    """按 collection_name 构建一条完整 chain"""
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=db_path,
    )
    retriever = db.as_retriever(search_kwargs={"k": 30})
    #构建有记忆的检索器（水龙头）
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, transform_prompt
    )
    #构建回答问题所用的系统提示词
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
    #将rag_chain包装成可运行的chain，并且自动添加会话历史
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

def _chain_key(collection_name: str, prompt_id: int, temperature: float) -> str:
    return f"{collection_name}::p{prompt_id}::t{temperature}"

def get_chain(collection_name: str,qa_system_prompt: str, prompt_id: int, temperature: float) -> RunnableWithMessageHistory:
    key = _chain_key(collection_name, prompt_id, temperature)
    if key not in _chain_cache:
        _chain_cache[key] = _build_chain(collection_name, qa_system_prompt, temperature)
    return _chain_cache[key]
#为了让FastAPI能够直接调用，需要把final_chain封装起来
class RAGService:
    #获取会话历史
    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        return get_session_history(session_id)
    #异步生成器
    async def chat(
        self, 
        query: str, 
        session_id: str, 
        collection_name: str, 
        qa_system_prompt: str, 
        prompt_id: int, 
        temperature: float,
        is_voice_mode: bool,
        ):
        chain = get_chain(
            collection_name, 
            qa_system_prompt,
            prompt_id,
            temperature,
            )
        #遍历大模型吐出的数据块
        async for chunk in chain.astream(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        ):
            #找到带有answer标签的推送到调用端
            if "answer" in chunk:
                yield chunk["answer"]