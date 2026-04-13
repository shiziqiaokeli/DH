import json
import pytest
import asyncio
from pathlib import Path
from .eval_runner import (
    run_rag_batch,
    run_ragas_eval,
    run_deepeval_eval,
    TESTSET_PATH,
    BASELINE_PATH,
)

THRESHOLD_DROP = 0.05

@pytest.fixture(scope="module")
def testset():
    if not TESTSET_PATH.exists():
        pytest.skip("测试集不存在，跳过评测")
    return json.loads(TESTSET_PATH.read_text(encoding="utf-8"))

@pytest.fixture(scope="module")
def baseline():
    if not BASELINE_PATH.exists():
        pytest.skip("基线文件不存在，请先运行 --update-baseline")
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

@pytest.fixture(scope="module")
def rag_results(testset):
    return asyncio.get_event_loop().run_until_complete(
        run_rag_batch(testset)
    )

class TestRAGContextQuality:   
    def test_ragas_context_precision(self, rag_results, baseline):   #检索精度
        scores = run_ragas_eval(rag_results)
        base = baseline.get("ragas_context_precision", 0)
        assert scores["ragas_context_precision"] >= base - THRESHOLD_DROP, (
            f"上下文精度下降: {scores['ragas_context_precision']:.4f} "
            f"< 基线 {base:.4f}"
        )
    def test_ragas_context_recall(self, rag_results, baseline):   #检索召回率
        scores = run_ragas_eval(rag_results)
        base = baseline.get("ragas_context_recall", 0)
        assert scores["ragas_context_recall"] >= base - THRESHOLD_DROP

class TestRAGFaithfulness:   
    def test_ragas_faithfulness(self, rag_results, baseline):   #忠实度
        scores = run_ragas_eval(rag_results)
        base = baseline.get("ragas_faithfulness", 0)
        assert scores["ragas_faithfulness"] >= base - THRESHOLD_DROP
    def test_deepeval_hallucination(self, rag_results, baseline):   #幻觉率
        scores = run_deepeval_eval(rag_results)
        base = baseline.get("deepeval_hallucination", 0)
        assert scores["deepeval_hallucination"] >= base - THRESHOLD_DROP