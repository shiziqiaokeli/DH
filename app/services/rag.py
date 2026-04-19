'''本地模型
conda activate vllm
python3 -m vllm.entrypoints.openai.api_server \
    --model /mnt/d/huggingface/Qwen2.5-3B-AWQ \
    --port 9526 \
    --quantization awq \
    --gpu-memory-utilization 0.4 \
    --max-model-len 4096 \
    --kv-cache-dtype fp8 \
    --dtype float16 \
    --host 0.0.0.0
'''
from app.core.config import settings#导入配置文件
from opentelemetry import trace#导入trace模块
from app.core.custom_embed import CustomQwenEmbeddings#导入自定义向量模型（api）
from langchain_huggingface import HuggingFaceEmbeddings#导入向量模型（本地）
from langchain_community.cross_encoders import HuggingFaceCrossEncoder#导入重排器模型
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker#导入重排器
from langchain_community.document_loaders import TextLoader#初始化加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter#初始化分割器
import uuid#生成唯一uuid
from langchain_chroma import Chroma#初始化向量数据库
import asyncio#异步核心
from langchain_openai import ChatOpenAI#初始化大模型
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder#导入提示词拼接工具和记忆区模块
from langchain_community.chat_message_histories import RedisChatMessageHistory#导入Redis会话历史模块
from langchain_core.documents import Document#导入Document对象
from langchain_community.retrievers import BM25Retriever#导入BM25检索器
from langchain_classic.retrievers import EnsembleRetriever,ContextualCompressionRetriever#导入多路融合检索器，上下文压缩检索器
from langchain_classic.chains import create_history_aware_retriever#构建有记忆的检索器（水龙头）
from langchain_classic.chains.combine_documents import create_stuff_documents_chain#接受前面所有chain的输出作为context（排水口）
from langchain_classic.chains import create_retrieval_chain#规定两个chain的协议（水管）
from langchain_core.runnables import RunnableWithMessageHistory#将rag_chain包装成可运行的chain，并且自动添加会话历史
from langchain_core.chat_history import BaseChatMessageHistory#添加langchain标准的数据接口
import time
from app.observability.metrics import (
    llm_requests_total, llm_errors_total, llm_request_duration,
    llm_tokens_total, llm_tokens_per_request,
    rag_retrieved_docs, rag_retrieval_top_score, llm_output_chars,
)#导入自定义LLM/RAG业务指标

db_path = settings.VECTOR_DB_PATH   #获取向量数据库路径
_tracer = trace.get_tracer("dh.rag")

'''
embeddings = CustomQwenEmbeddings(   #初始化向量模型（api）
    api_key=settings.LLM_API_KEY,
    base_url=settings.LLM_BASE_URL,
    model="text-embedding-v3",
)
'''
embeddings = HuggingFaceEmbeddings(   #初始化向量模型（本地）
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},   #有NVIDIA GPU改为"cuda"（GPU吃紧，先用CPU）
    encode_kwargs={"normalize_embeddings": True},
)


embeddings._client.half()   #转fp16，显存吃紧
_reranker_model = HuggingFaceCrossEncoder(   #初始化重排器模型
    model_name="BAAI/bge-reranker-v2-m3",
    model_kwargs={"device": "cuda"},   #有NVIDIA GPU改为"cuda"（GPU吃紧，先用CPU）
)
_reranker_model.client.model.half()   #转fp16，显存吃紧
reranker = CrossEncoderReranker(model=_reranker_model, top_n=6)   #初始化重排器

async def process_uploaded_file(file_path: str) -> str:   #将上传的文件用向量模型切分并存入到向量数据库
    loader = TextLoader(file_path, encoding="utf-8")   #通过路径和编码方式初始化加载器
    documents = loader.load()   #调用加载器的load方法获取资料
    splitter = RecursiveCharacterTextSplitter(   #初始化分割器
        chunk_size=400,   #每块包含的最大字符数
        chunk_overlap=100,   #相邻两块之间重叠的字符数
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ",", " ", ""],   #分隔符（控制切分优先级，从左到右依次切分，直到chunk_size<300）
    )
    chunks = splitter.split_documents(documents)   #调用分割器的split_documents方法分割资料
    collection_name = f"kb_{uuid.uuid4().hex[:12]}"   #生成唯一uuid作为collection_name
    db = Chroma(   #初始化向量数据库
        collection_name=collection_name,   
        embedding_function=embeddings,
        persist_directory=db_path,
    )
    await asyncio.to_thread(db.add_documents, chunks)   #开启一个独立线程等待向量数据库将资料存入，期间CPU可以处理其他请求
    return collection_name   #返回给调用方，写入MySQL


def _make_llm(temperature: float) -> ChatOpenAI:   #初始化大模型（api）
    return ChatOpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model="qwen3.6-flash",
        temperature=temperature,
        streaming=True,
        stream_usage=True, 
    )
'''
def _make_llm(temperature: float) -> ChatOpenAI:   #初始化大模型（本地）
    return ChatOpenAI(
        api_key="EMPTY",
        base_url="http://localhost:9526/v1",
        model="/mnt/d/huggingface/Qwen2.5-3B-AWQ",
        temperature=temperature,
        streaming=True,
    )
'''

transform_system_prompt=(   #构建转化问题所用的系统提示词
    "给定聊天历史和最新用户问题，"
    "该用户问题可能引用了聊天历史中的上下文（比如代词'它'）。"
    "请构建一个可以脱离聊天历史独立理解的全新问题。"
    "你只需要重写问题，如果不需要重写就原样返回，绝对不要尝试回答它。"
)

transform_prompt = ChatPromptTemplate.from_messages([   #把系统提示词，记忆区，用户问题拼接起来
    ("system", transform_system_prompt),  
    MessagesPlaceholder("chat_history"),   #记忆区，0轮时为空，每一轮对话后会自动添加到记忆区
    ("human", "{input}"),  
])

redis_url=settings.REDIS_URL   #获得Redis的URL

def get_session_history(session_id: str):   #根据session_id获取会话历史
    return RedisChatMessageHistory(
        session_id=session_id,
        url=redis_url,
    )

def _build_chain(   #构建完整的chain
    collection_name: str,
    qa_system_prompt: str, 
    temperature: float
    ) -> RunnableWithMessageHistory:   
    llm = _make_llm(temperature)
    db = Chroma(   #稠密检索：Chroma向量库
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=db_path,
    )
    dense_retriever = db.as_retriever(search_kwargs={"k": 20})  
    raw = db.get(include=["documents", "metadatas"])   #稀疏检索：从Chroma取出全部原始文本，构建 BM25
    bm25_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs, k=20)
    ensemble_retriever = EnsembleRetriever(   #多路融合：RRF（倒数排名融合），权重各0.5
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.4, 0.6],
    )
    compression_retriever = ContextualCompressionRetriever(   #精排：BGE-Reranker压缩到top-5
        base_compressor=reranker,
        base_retriever=ensemble_retriever,
    )
    history_aware_retriever = create_history_aware_retriever(   #构建有记忆的检索器（水龙头）
        llm, 
        compression_retriever, 
        transform_prompt,   #把transform_prompt喂给llm得到转化后的问题，把转化后的问题通过retriever先向量化再检索出相似度最高的k个资料传给下一个chain
    )
    qa_prompt = ChatPromptTemplate.from_messages([   #构建回答问题所用的提示词，把系统提示词，记忆区，用户问题拼接起来
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)   #接受前面所有chain的输出作为context（排水口），赋值给qa_prompt，然后调用llm回答问题
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)   #规定两个chain的协议（水管）
    return RunnableWithMessageHistory(   #将rag_chain包装成可运行的chain，并且自动添加会话历史
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

_chain_cache: dict[str, RunnableWithMessageHistory] = {}   #初始化chain缓存

def _chain_key(collection_name: str, prompt_id: int, temperature: float) -> str:   #构建chain的唯一标识
    return f"{collection_name}::p{prompt_id}::t{temperature}"

def get_chain(   #获取chain（避免重复创建昂贵的chain对象，从而提高程序的运行速度并节省内存）
    collection_name: str,
    qa_system_prompt: str, 
    prompt_id: int, 
    temperature: float,
    ) -> RunnableWithMessageHistory:
    key = _chain_key(collection_name, prompt_id, temperature)   #构建chain的唯一标识
    if key not in _chain_cache:   
        _chain_cache[key] = _build_chain(collection_name, qa_system_prompt, temperature)   #如果chain不存在，则创建
    return _chain_cache[key]

class RAGService:
    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        return get_session_history(session_id)   #获取会话历史
    async def chat(
        self, 
        query: str, 
        session_id: str, 
        collection_name: str, 
        qa_system_prompt: str, 
        prompt_id: int, 
        temperature: float,
        is_voice_mode: bool,   #main函数有校验，不能删除
    ):
        attrs = {   #统一span/metric标签
            "session.id": session_id,
            "rag.collection_name": collection_name,
            "rag.prompt_id": prompt_id,
            "gen_ai.request.temperature": temperature,
            "app.voice_mode": is_voice_mode,
        }
        llm_requests_total.add(1, attrs)
        t_start = time.perf_counter()
        t_first_token: float | None = None
        output_buf: list[str] = []
        with _tracer.start_as_current_span("rag.chat", attributes=attrs) as span:
            try:
                chain = get_chain(collection_name, qa_system_prompt, prompt_id, temperature)
                async for chunk in chain.astream(   #遍历大模型吐出的数据块
                    {"input": query},
                    config={"configurable": {"session_id": session_id}},
                ):
                    if "answer" in chunk and chunk["answer"]:
                        if t_first_token is None:   #首token到达，记录TTFT
                            t_first_token = time.perf_counter()
                            ttft = t_first_token - t_start
                            span.add_event("first_token", {"ttft_seconds": ttft})
                            llm_request_duration.record(ttft, {**attrs, "stage": "ttft"})
                        output_buf.append(chunk["answer"])
                        yield chunk["answer"]   #找到带有answer标签的推送到调用端
                    if "answer" in chunk and hasattr(chunk.get("answer"), "usage_metadata"):
                        usage = chunk["answer"].usage_metadata or {}
                        in_tok, out_tok = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                        if in_tok:
                            llm_tokens_total.add(in_tok, {**attrs, "direction": "input"})
                            llm_tokens_per_request.record(in_tok, {**attrs, "direction": "input"})
                        if out_tok:
                            llm_tokens_total.add(out_tok, {**attrs, "direction": "output"})
                            llm_tokens_per_request.record(out_tok, {**attrs, "direction": "output"})
                full_out = "".join(output_buf)   #流结束：记录E2E延迟、输出长度
                e2e = time.perf_counter() - t_start
                llm_request_duration.record(e2e, {**attrs, "stage": "e2e"})
                llm_output_chars.record(len(full_out), attrs)
                span.set_attribute("gen_ai.completion.length", len(full_out))
            except Exception as e:
                llm_errors_total.add(1, {**attrs, "error.type": type(e).__name__})
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise  