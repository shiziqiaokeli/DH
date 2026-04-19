"""
OpenTelemetry初始化：Tracer+Meter+自动埋点
端口      容器/进程      协议                 角色                 谁主动连他
4317      Phoenix       gRPC   OTLP Trace接收                    FastAPI应用
6006      Phoenix       HTTP   Phoenix UI（看trace）               浏览器
9464      FastAPI       HTTP   Prometheus Exporter（暴露metric）  Prometheus
9091      Pushgateway   HTTP   离线指标暂存           eval_runner（push）+Prometheus（pull）
9090      Prometheus    HTTP   TSDB查询+告警评估                Grafana、浏览器
9093      Alertmanager  HTTP   告警去重/路由/通知          Prometheus（push）、浏览器
3000      Grafana       HTTP   统一可视化大屏                       浏览器
http://localhost:9091   #Pushgateway临时中转站
http://localhost:9090   #Prometheus数据库（Metric）
http://localhost:3000   #Grafana数据库图表化（URL：http://prometheus:9090）
http://localhost:6006   #Phoenix（链路追踪Trace）
http://localhost:9093   #Alertmanager警报系统
docker compose -f docker-compose.observability.yml up -d   #启动
docker compose -f docker-compose.observability.yml ps   #查看状态
docker compose -f docker-compose.observability.yml stop   #保留数据卷
docker compose -f docker-compose.observability.yml down -v   #完全清理  
"""
from __future__ import annotations
import os
from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor
from prometheus_client import start_http_server

SERVICE_NAME = "dh-multimodal-engine"
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.1.0")
#4317默认端口，gRPC协议，速度快、延迟低、二进制传输，适合生产环境和后端服务
#4318端口，HTTP/JSON协议，兼容性好，适合前端、浏览器插件或无法使用gRPC的受限环境
OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")   
PROM_PORT = int(os.getenv("PROM_EXPORTER_PORT", "9464"))
os.environ.setdefault("OPENINFERENCE_HIDE_INPUTS", "false")
os.environ.setdefault("OPENINFERENCE_HIDE_OUTPUTS", "false")
os.environ.setdefault("OPENINFERENCE_HIDE_INPUT_TEXT", "false")

_initialized = False

def init_observability(app, engine=None) -> None:   #FastAPI启动时调用一次；engine是SQLAlchemy的AsyncEngine
    global _initialized
    if _initialized:
        return
    resource = Resource.create({
        "service.name": SERVICE_NAME,
        "service.version": SERVICE_VERSION,
        "deployment.environment": os.getenv("DEPLOY_ENV", "dev"),
    })
    tracer_provider = TracerProvider(resource=resource)   #Trace：OTLP到Phoenix/Tempo
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True))
    )
    trace.set_tracer_provider(tracer_provider)
    prom_reader = PrometheusMetricReader()   #Metric：同时走Prometheus（pull）和OTLP（push）
    meter_provider = MeterProvider(resource=resource, metric_readers=[prom_reader])
    metrics.set_meter_provider(meter_provider)
    start_http_server(PROM_PORT)   #开Prometheus的/metrics HTTP端点
    FastAPIInstrumentor.instrument_app(app)   #自动埋点：FastAPI路由/Redis/httpx/SQLAlchemy
    HTTPXClientInstrumentor().instrument()
    RedisInstrumentor().instrument()
    if engine is not None:
        SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)
    LoggingInstrumentor().instrument(set_logging_format=True)
    LangChainInstrumentor().instrument()   #关键：LangChain全链路自动span+token统计
    _initialized = True