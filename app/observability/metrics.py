"""
自定义LLM/RAG业务指标：请求数、延迟（分阶段）、token、错误率、输出长度分布
"""
from __future__ import annotations
from opentelemetry import metrics

_meter = metrics.get_meter("dh.llm")

llm_requests_total = _meter.create_counter(   #请求数
    name="llm_requests_total",
    description="LLM请求总数",
    unit="1",
)

llm_errors_total = _meter.create_counter(   #错误数（按错误类型打标）
    name="llm_errors_total",
    description="LLM请求错误数",
    unit="1",
)

llm_request_duration = _meter.create_histogram(   #分阶段延迟（stage：e2e/retrieval/rerank/llm/ttft）
    name="llm_request_duration_seconds",
    description="LLM请求延迟（按阶段）",
    unit="s",
)

llm_tokens_total = _meter.create_counter(   #token总数（direction：input/output，必须分开）
    name="llm_tokens_total",
    description="LLM累计token消耗",
    unit="1",
)

llm_tokens_per_request = _meter.create_histogram(   #单次请求token分布
    name="llm_tokens_per_request",
    description="单次请求token数分布",
    unit="1",
)

rag_retrieved_docs = _meter.create_histogram(   #检索召回文档数分布
    name="rag_retrieved_docs",
    description="RAG检索召回文档数",
    unit="1",
)

rag_retrieval_top_score = _meter.create_histogram(   #Top1重排分数分布（用于监测检索质量漂移）
    name="rag_retrieval_top_score",
    description="RAG重排Top1分数",
    unit="1",
)

llm_output_chars = _meter.create_histogram(   #输出字符数分布（长尾突变→drift信号）
    name="llm_output_chars",
    description="LLM输出字符长度",
    unit="1",
)

eval_score = _meter.create_gauge(   #离线评测指标（Gauge，由eval_runner推送）
    name="llm_eval_score",
    description="离线评测分数（幻觉率/忠实度等）",
    unit="1",
)