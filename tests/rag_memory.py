from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.custom_embed import CustomQwenEmbeddings
from app.core.config import settings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

base_dir=Path(__file__).resolve().parent.parent
file_path=base_dir / "data" / "njust_info.txt"
textloader=TextLoader(file_path,encoding="utf-8")
documents=textloader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks=text_splitter.split_documents(documents)
embeddings=CustomQwenEmbeddings(
    api_key=settings.LLM_API_KEY,
    base_url=settings.LLM_BASE_URL,
    model="text-embedding-v1"
    )
db=Chroma.from_documents(
    chunks, 
    embeddings, 
    persist_directory="./memory_chroma_db"
    )
retriever=db.as_retriever(search_kwargs={"k": 3})
transform_system_prompt=(
    "给定聊天历史和最新用户问题，"
    "该用户问题可能引用了聊天历史中的上下文（比如代词'它'）。"
    "请构建一个可以脱离聊天历史独立理解的全新问题。"
    "你只需要重写问题，如果不需要重写就原样返回，绝对不要尝试回答它。"
)
transform_prompt = ChatPromptTemplate.from_messages([
    ("system", transform_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
llm=ChatOpenAI(
    api_key=settings.LLM_API_KEY,
    base_url=settings.LLM_BASE_URL,
    model="qwen3.5-flash",
    temperature=0.1,
    streaming=True
    )
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, transform_prompt
)
qa_system_prompt="""你是一名知识渊博的南京理工大学（NJUST）资深校友。
请根据提供的【校史参考资料】来回答【校友提问】。

要求：
1. 仅根据资料内容回答，不要胡编乱造。
2. 如果资料中没提到相关信息，请礼貌地回答：“抱歉，这段校史我还需要再查证一下。”
3. 回答语气要亲切、严谨，体现出南理工“献身”精神的底蕴。

【校史参考资料】：
{context}"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
store={}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
final_chain=RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",          
    history_messages_key="chat_history", 
    output_messages_key="answer",       
)
async def main():
    queries = [
        "中国人民解放军炮兵工程学院哪年成立？",
        "那它后来改名了吗？" # 故意省略主语，测试记忆
    ]
    for query in queries:
        print(f"校友提问: {query}\n")
        print("资深校友正在回忆中...")
        # 注意：现在传给 astream 的input必须是一个字典 {"input": query}
        # 同时config也是一个字典{"configurable": {"session_id": "default"}}
        async for chunk in final_chain.astream({"input": query},config={"configurable": {"session_id": "default"}}):
            if "answer" in chunk:
                print(chunk["answer"], end="", flush=True)
        print("\n\n---回答结束---")
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())




