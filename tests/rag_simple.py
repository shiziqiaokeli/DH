from app.core.config import settings  # 导入全局配置对象
import asyncio                      # 导入异步支持模块
from openai import AsyncOpenAI      # 导入 OpenAI 异步客户端
from pathlib import Path
import re
import chromadb
from chromadb.config import Settings

base_dir = Path(__file__).resolve().parent.parent
file_path = base_dir / "data" / "njust_info.txt"
content = file_path.read_text(encoding="utf-8") #读取文件内容

#将长字符串切分成短文本列表
def split_text(
    text: str, #要切分的文本
    chunk_size: int = 300, #每个短文本的长度
    separator: str = r"\n\n|。" #分隔符（段落间的空行或者中文句号）
    ) -> list[str]:
    #使用正则表达式将文本切分成短文本列表
   
   initial_chunks = re.split(separator, text)
   final_chunks = []
   current_chunk = ""

   for chunk in initial_chunks:
      chunk=chunk.strip()
      if not chunk:
        continue
      if len(current_chunk) + len(chunk) <= chunk_size:
          current_chunk += (" "+chunk if current_chunk else chunk)
      else:
          if current_chunk:
              final_chunks.append(current_chunk)
          if len(chunk) > chunk_size:
            for i in range(0, len(chunk), chunk_size):
                final_chunks.append(chunk[i:i+chunk_size])
            current_chunk = ""
          else:
            current_chunk = chunk
   if current_chunk:
        final_chunks.append(current_chunk)

   return final_chunks

client = AsyncOpenAI(
    api_key=settings.LLM_API_KEY,
    base_url=settings.LLM_BASE_URL,
)

async def embedding(text: list[str]):
    response =  await client.embeddings.create(
        model="text-embedding-v1",
        input=text,
    )
    return [item.embedding for item in response.data]

async def ask_njust_question(question:str,n:int,collection):
    #问题向量化
    query_response= await client.embeddings.create(
        model="text-embedding-v1",
        input=question,
    )
    #返回向量
    query_vector=query_response.data[0].embedding
    #通过私有知识库寻找最相似的文本
    result=collection.query(
        query_embeddings=[query_vector],
        n_results=n,
    )
    #拼接n段文本
    current_docs=result["documents"][0]
    final_docs="\n\n".join([f"资料片段{i+1}:{doc}"for i,doc in enumerate(current_docs)])
    #构建提示词
    system_prompt = "你是一个专业的南理工智能客服。你必须保持严谨、客观，且说话得体。"
    
    final_prompt = f"""
### 角色指令
你是一个基于私有知识库的智能客服。请严格根据以下给出的【参考资料】来回答【用户问题】。

### 约束规则
1. 如果【参考资料】中没有提到相关信息，请直接回答：“抱歉，在我的知识库中没有找到关于此问题的记录，无法为您提供准确回答。”
2. 严禁编造任何事实，严禁使用模型自带的外部知识来补充参考资料中缺失的部分。
3. 回答排版请使用清晰的 Markdown 格式。

### 【参考资料】
{final_docs}

### 【用户问题】
{question}

---
请开始你的回答：
"""
    #调用大模型回答问题
    try:
        # 调用 openai 聊天接口，使用 stream=True 实现流式输出，降低首包延迟
        response = await client.chat.completions.create(
            model="qwen3.5-flash",
            messages=[
                {"role": "system", "content":system_prompt },
                {"role": "user", "content":final_prompt }
            ],
            temperature=0.1,
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

async def main():

    chunks = split_text(content, chunk_size=100)
    
    #embeddings = await embedding(chunks)#文本向量化
    
    client_db = chromadb.PersistentClient(path="./chroma_db")

    collection = client_db.get_or_create_collection(name="njust_history")

    '''
    存入向量数据库
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))],
    )
    '''
    '''
    打印向量
    for i, c in enumerate(embeddings):
        print(f"段落 {i+1} : {c[:5]}...")
    '''
    await ask_njust_question("信息自动化与制造工程学院在哪年成立？",3,collection)
    
if __name__ == "__main__":
    asyncio.run(main())