#RAG评测流水线（Ragas+DeepEval）
#首次使用：python -m tests.eval.eval_runner --update-baseline
#后续使用：python -m tests.eval.eval_runner
'''
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
'''
from pathlib import Path
from deepeval.models import DeepEvalBaseLLM#自定义裁判LLM
from openai import OpenAI, AsyncOpenAI
from app.core.config import settings
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
import asyncio
from app.services.rag import (  #复用项目内的RAG组件
    _build_chain,
    get_session_history,
)
import math
from datasets import Dataset#RAGAS评测数据集
from ragas import evaluate as ragas_evaluate#评测
from ragas.metrics import (#RAGAS评测指标
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig#RAGAS评测配置
import os#控制环境变量
from deepeval.test_case import LLMTestCase#DeepEval评测用例
from deepeval import evaluate as deepeval_evaluate#DeepEval评测
from deepeval.metrics import (#DeepEval评测指标
    HallucinationMetric,
    ContextualPrecisionMetric,
    AnswerRelevancyMetric,
)
from deepeval.evaluate import AsyncConfig#DeepEval评测配置
import sys
import app.services.rag as rag_module
import gc
import torch
import json

EVAL_DIR = Path(__file__).parent
TESTSET_PATH = EVAL_DIR / "testset.json"
BASELINE_PATH = EVAL_DIR / "baseline.json"
EVAL_QA_PROMPT = (   #评测用的QA系统提示词（与生产一致）
    "你是一名精通《三国演义》的学者。"
    "请阅读以下提供的【参考资料】，然后仅根据原文内容回答问题。"
    "如果【参考资料】是诗词或赞文，请提取其中隐含的事实。"
    "若【参考资料】未提及，请回答不知道，不要引用民间传说（如三国演义与三国志的区别）。"
    "【参考资料】：{context}"
)

class QwenJudge(DeepEvalBaseLLM):   #将本地Qwen模型包装成裁判LLM
    name = "qwen-plus-2025-07-28"
    def __init__(self):
        self._client = OpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
        )
        self._async_client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
        )
        super().__init__(self.name)
    def load_model(self):
        return self._client
    def generate(self, prompt: str, *args, **kwargs) -> str:
        resp = self._client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content
    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        resp = await self._async_client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content
    def get_model_name(self) -> str:
        return self.name

eval_judge_llm = LangchainLLMWrapper(ChatOpenAI(   #初始化裁判LLM
    api_key=settings.LLM_API_KEY,
    base_url=settings.LLM_BASE_URL,
    model=QwenJudge.name,
    temperature=0,
    n=1,
    max_retries=3,
))
eval_judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(   #初始化裁判向量模型
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
))
'''
async def run_rag_batch(   #批量调用RAG链，收集question/answer/contexts/ground_truth
    testset: list[dict],   #并发太多，爆显存了
    temperature: float = 0.1,
    ) -> list[dict]:
    results = []
    for i, item in enumerate(testset):   
        chain = _build_chain(
            collection_name=item["collection_name"],
            qa_system_prompt=EVAL_QA_PROMPT,
            temperature=temperature,
        )
        eval_session_id = f"__eval_session_{i}"
        history = get_session_history(eval_session_id)
        await history.aclear()   #清除评测用的临时会话历史   
        response = await chain.ainvoke(   #调用链路，收集完整响应
            {"input": item["question"]},
            config={"configurable": {"session_id": eval_session_id}},
        )
        answer_text = response.get("answer", "")
        context_docs = response.get("context", [])   #提取检索到的上下文
        contexts = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in context_docs
        ]
        results.append({
            "question": item["question"],
            "answer": answer_text,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        })
        await history.aclear()   #清理临时会话
    return results
'''
MAX_CONCURRENCY = 1   #Ollama本地推理的安全并发数，根据显存调整
BATCH_COOLDOWN = 1.0   #每批之间冷却秒数，防止GPU过热/OOM

async def _run_single(   #单条测试用例的推理，受信号量限流
    semaphore: asyncio.Semaphore,
    item: dict,
    index: int,
    temperature: float,
) -> dict:
    async with semaphore:
        chain = _build_chain(
            collection_name=item["collection_name"],
            qa_system_prompt=EVAL_QA_PROMPT,
            temperature=temperature,
        )
        eval_session_id = f"__eval_session_{index}"
        history = get_session_history(eval_session_id)
        await history.aclear()
        response = await chain.ainvoke(
            {"input": item["question"]},
            config={"configurable": {"session_id": eval_session_id}},
        )
        answer_text = response.get("answer", "")
        context_docs = response.get("context", [])
        contexts = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in context_docs
        ]
        await history.aclear()
        print(f"  [{index + 1}/{len(item.get('_total', []))}] 完成: {item['question'][:30]}...")
        return {
            "question": item["question"],
            "answer": answer_text,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        }

async def run_rag_batch(
    testset: list[dict],
    temperature: float = 0.1,
) -> list[dict]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [
        _run_single(semaphore, item, i, temperature)
        for i, item in enumerate(testset)
    ]
    results = []
    batch_size = MAX_CONCURRENCY * 2    #分批收集，避免一次性全部gather导致内存峰值过高
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        for r in batch_results:
            if isinstance(r, Exception):
                print(f"  [ERROR] {r}")
                continue
            results.append(r)
        if start + batch_size < len(tasks):
            await asyncio.sleep(BATCH_COOLDOWN)
    return results

def _safe_mean(values) -> float:   #对RAGAS返回的列表求均值，跳过NaN
    if isinstance(values, (int, float)):
        return float(values)
    valid = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return sum(valid) / len(valid) if valid else 0.0

def run_ragas_eval(results: list[dict]) -> dict[str, float]:   #使用RAGAS框架评测
    ds = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })
    score = ragas_evaluate(
        ds,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
        ],
        llm=eval_judge_llm,
        embeddings=eval_judge_embeddings,
        run_config=RunConfig(max_workers=1, max_wait=300),  # 限制并发
    )
    return {
        "ragas_context_precision": _safe_mean(score["context_precision"]),  #检索内容精度
        "ragas_context_recall": _safe_mean(score["context_recall"]),   #检索内容召回率
        "ragas_faithfulness": _safe_mean(score["faithfulness"]),   #忠实度
    }

os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "YES"   #通过环境变量，彻底禁用DeepEval框架的超时限制

def run_deepeval_eval(results: list[dict]) -> dict[str, float]:   #使用DeepEval框架评测
    test_cases = []
    for r in results:
        tc = LLMTestCase(
            input=r["question"],
            actual_output=r["answer"],
            expected_output=r["ground_truth"],   
            context=r["contexts"],   
            retrieval_context=r["contexts"],   
        )
        test_cases.append(tc)
    deepeval_judge = QwenJudge()
    hallucination_metric = HallucinationMetric(
        threshold=0.5,
        model=deepeval_judge,   
    )
    contextual_precision_metric = ContextualPrecisionMetric(
        threshold=0.5,
        model=deepeval_judge,
    )
    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.5,
        model=deepeval_judge,
    )
    eval_result = deepeval_evaluate(
        test_cases=test_cases,
        metrics=[hallucination_metric, contextual_precision_metric, answer_relevancy_metric],
        async_config=AsyncConfig(max_concurrent=1),   # 限制并发为 1，防止 API 限流
    )
    hall_scores = []
    cp_scores = []
    ar_scores = []
    for tr in eval_result.test_results:
        for md in tr.metrics_data:
            if md.name == "Hallucination":
                if md.score is not None:
                    hall_scores.append(md.score)
            elif md.name == "Contextual Precision":
                if md.score is not None:
                    cp_scores.append(md.score)
            elif md.name == "Answer Relevancy":
                if md.score is not None:
                    ar_scores.append(md.score)
    return {
        "deepeval_hallucination": sum(hall_scores) / len(hall_scores) if hall_scores else 0.0,   #幻觉率
        "deepeval_contextual_precision": sum(cp_scores) / len(cp_scores) if cp_scores else 0.0,   #检索内容精度
        "deepeval_answer_relevancy": sum(ar_scores) / len(ar_scores) if ar_scores else 0.0,   #答案相关性
    }

def check_regression(   #对比基线，返回 (是否通过, 失败原因列表)
    current: dict[str, float],
    baseline: dict[str, float],
) -> tuple[bool, list[str]]:
    threshold_drop = baseline.get("threshold_drop", 0.05)
    failures = []
    for key, base_val in baseline.items():
        if key == "threshold_drop":
            continue
        curr_val = current.get(key)
        if curr_val is None:
            continue
        if curr_val < base_val - threshold_drop:
            failures.append(
                f"  {key}: {curr_val:.4f} < 基线 {base_val:.4f} "
                f"(容差 {threshold_drop})"
            )
    return len(failures) == 0, failures

async def main():
    update_baseline = "--update-baseline" in sys.argv
    if not TESTSET_PATH.exists():   #加载测试集
        print(f"错误: 未找到测试集 {TESTSET_PATH}")
        print("请先创建黄金测试集（至少 20 条 Q&A）")
        sys.exit(1)
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    print(f"已加载 {len(testset)} 条测试用例")
    print("正在批量调用 RAG 链路...")   #批量跑RAG
    results = await run_rag_batch(testset)
    print(f"已完成 {len(results)} 条推理")
    rag_module._chain_cache.clear()       # 清除链缓存（内部持有 Chroma/embeddings 引用）
    del rag_module.embeddings
    del rag_module._reranker_model
    del rag_module.reranker
    gc.collect()  
    torch.cuda.empty_cache()
    print("已释放 bge-m3 和 bge-reranker-v2-m3 显存")
    print("正在运行 RAGAS 评测...")   #RAGAS评测
    ragas_scores = run_ragas_eval(results)
    for k, v in ragas_scores.items():
        print(f"  {k}: {v:.4f}")
    print("正在运行 DeepEval 评测...")   #DeepEval评测
    deepeval_scores = run_deepeval_eval(results)
    for k, v in deepeval_scores.items():
        print(f"  {k}: {v:.4f}")
    all_scores = {**ragas_scores, **deepeval_scores}   #合并分数
    if update_baseline or not BASELINE_PATH.exists():   #首次运行或强制更新基线
        all_scores["threshold_drop"] = 0.05
        BASELINE_PATH.write_text(
            json.dumps(all_scores, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n基线已保存到 {BASELINE_PATH}")
        sys.exit(0)
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))   #回归检测
    passed, failures = check_regression(all_scores, baseline)
    if passed:
        print("\n所有指标通过回归检测！")
        sys.exit(0)
    else:
        print("\n指标回归检测失败：")
        for f in failures:
            print(f)
        print("\n本次 push 被拒绝。请修复 RAG 质量后重试。")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())