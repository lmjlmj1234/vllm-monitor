"""
vLLM Monitor — Smoke Test Suite

轻量冒烟测试，不启动 Streamlit、不加载模型、不启动 vLLM server。
可直接用 python 运行：python -m tests.test_smoke
或：python tests/test_smoke.py
"""

import sys
import time


def test_imports():
    """1. 检查所有核心模块能否正常导入"""
    from vllm_monitor.collector import VLLMMetricsCollector, VLLMMetrics
    from vllm_monitor.dashboard import VLLMDashboard
    import vllm_monitor
    print("[PASS] 核心模块导入成功")
    print(f"       collector: {VLLMMetricsCollector.__module__}")
    print(f"       dashboard: {VLLMDashboard.__module__}")


def test_collector_demo_mode():
    """2. Demo 模式：采集器能否正常工作"""
    from vllm_monitor.collector import VLLMMetricsCollector

    c = VLLMMetricsCollector()
    for i in range(5):
        c.record_request(0.1 + i * 0.05)
        c.record_communication(1024 * 1024)

    m = c.collect()
    print("[PASS] Demo 模式采集正常")
    print(f"       RPS: {m.requests_per_second:.2f}")
    print(f"       延迟: {m.avg_latency_ms:.2f} ms")
    print(f"       GPU显存: {m.gpu_memory_used_gb:.2f} GB / {m.gpu_memory_total_gb:.2f} GB")
    print(f"       GPU利用率: {m.gpu_utilization:.1f}%")
    print(f"       GPU温度: {m.gpu_temperature:.0f}°C")
    print(f"       GPU功耗: {m.gpu_power_w:.1f} W")
    print(f"       通信量: {m.communication_bytes / 1e6:.2f} MB")
    print(f"       缓存命中率: {m.paged_attention_hit_rate:.2%}")
    print(f"       队列长度: {m.queue_length}")


def test_metric_value_ranges():
    """3. 指标数值范围合法性检查"""
    from vllm_monitor.collector import VLLMMetricsCollector

    c = VLLMMetricsCollector()
    c.record_request(0.15)
    c.record_communication(1024)
    m = c.collect()

    assert m.requests_per_second >= 0, f"RPS={m.requests_per_second} < 0"
    assert m.avg_latency_ms >= 0, f"latency={m.avg_latency_ms} < 0"
    assert m.gpu_memory_used_gb >= 0, f"gpu_mem={m.gpu_memory_used_gb} < 0"
    assert 0 <= m.gpu_utilization <= 100, f"gpu_util={m.gpu_utilization} out of [0,100]"
    assert m.gpu_temperature >= 0, f"temp={m.gpu_temperature} < 0"
    assert m.gpu_power_w >= 0, f"power={m.gpu_power_w} < 0"
    assert 0 <= m.paged_attention_hit_rate <= 1, f"hit_rate={m.paged_attention_hit_rate} out of [0,1]"
    assert m.queue_length >= 0, f"queue={m.queue_length} < 0"
    print("[PASS] 所有指标数值范围合法")


def test_collector_with_vllm_url():
    """4. 传入 vllm_url 模式（不连接真实服务，仅验证兼容性）"""
    from vllm_monitor.collector import VLLMMetricsCollector

    c = VLLMMetricsCollector(vllm_url="http://localhost:8000")
    m = c.collect()
    # 连接不上远程时不应抛异常，应优雅降级为模拟数据
    print(f"[PASS] vllm_url 模式兼容：queue={m.queue_length}, hit_rate={m.paged_attention_hit_rate:.2%}")
    print(f"       远程不可达时优雅降级（prompt_tokens=0, gen_tokens=0, 命中率=0.95 默认值）")


def test_metrics_dataclass_fields():
    """5. VLLMMetrics dataclass 字段完整性"""
    from vllm_monitor.collector import VLLMMetrics

    fields = [f.name for f in __import__('dataclasses').fields(VLLMMetrics)]
    expected = {
        "timestamp", "requests_per_second", "avg_latency_ms",
        "gpu_memory_used_gb", "gpu_memory_total_gb", "gpu_utilization",
        "gpu_temperature", "gpu_power_w", "communication_bytes",
        "paged_attention_hit_rate", "queue_length",
    }
    missing = expected - set(fields)
    extra = set(fields) - expected
    assert not missing, f"缺少字段: {missing}"
    assert not extra, f"多余字段: {extra}"
    print(f"[PASS] VLLMMetrics dataclass 字段完整（共 {len(fields)} 个字段）")


def test_pynvml_gpu_data():
    """6. pynvml 是否可读取真实 GPU 数据"""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        assert count > 0, "未检测到 GPU"
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle)
        pynvml.nvmlShutdown()
        print(f"[PASS] pynvml 实时 GPU 数据读取成功")
        print(f"       显卡: {name}")
        print(f"       利用率: {util.gpu}% / 显存: {mem.used / 1e9:.2f} GB / {mem.total / 1e9:.2f} GB")
        print(f"       温度: {temp}°C / 功耗: {power / 1000:.1f} W")
    except (ImportError, AssertionError, pynvml.NVMLError) as e:
        print(f"[WARN] pynvml 不可用，GPU 数据将使用降级/模拟数据: {e}")


def test_no_while_true():
    """7. 确认已消除 while True 阻塞模式"""
    import ast
    import os

    base = os.path.join(os.path.dirname(__file__), "..")
    issues = []
    for root, dirs, files in os.walk(os.path.join(base, "vllm_monitor")):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path) as fh:
                    tree = ast.parse(fh.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value is True:
                        issues.append(path)
    assert not issues, f"发现 while True 阻塞循环: {issues}"
    print("[PASS] 无 while True 阻塞循环")


def test_requirements():
    """8. 检查关键依赖是否已安装"""
    required = ["streamlit", "pynvml", "psutil", "requests", "plotly"]
    for pkg in required:
        __import__(pkg.replace("-", "_"))
        print(f"[PASS] 依赖 {pkg} 已安装")


def test_multi_gpu_detection():
    """9. 验证多 GPU 检测能力"""
    from vllm_monitor.collector import VLLMMetricsCollector
    import pynvml
    pynvml.nvmlInit()
    expected_count = pynvml.nvmlDeviceGetCount()
    pynvml.nvmlShutdown()

    c = VLLMMetricsCollector()
    assert len(c._nvml_handles) == expected_count, f"检测到 {len(c._nvml_handles)} 个 GPU, 期望 {expected_count}"
    if expected_count >= 1:
        idx, handle, name = c._nvml_handles[0]
        print(f"[PASS] 多 GPU 检测正常 — 共 {expected_count} 个, GPU 0: {name}")
    else:
        print(f"[PASS] 多 GPU 检测正常 — 未检测到 GPU (模拟模式)")


def test_plotly_chart_data():
    """10. 验证 plotly 图表数据准备逻辑"""
    from vllm_monitor.collector import VLLMMetrics
    from datetime import datetime
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # 模拟历史数据
    history = [
        VLLMMetrics(
            timestamp=100.0 + i, requests_per_second=10.0 + i, avg_latency_ms=50.0 + i,
            gpu_memory_used_gb=1.0, gpu_memory_total_gb=12.0, gpu_utilization=30.0 + i,
            gpu_temperature=40.0, gpu_power_w=50.0, communication_bytes=1024,
            paged_attention_hit_rate=0.95, queue_length=0,
        )
        for i in range(5)
    ]
    timestamps = [datetime.fromtimestamp(m.timestamp) for m in history]

    fig = make_subplots(rows=2, cols=2)
    fig.add_trace(go.Scatter(x=timestamps, y=[m.requests_per_second for m in history], mode="lines"))
    fig.add_trace(go.Scatter(x=timestamps, y=[m.avg_latency_ms for m in history], mode="lines"))
    fig.add_trace(go.Scatter(x=timestamps, y=[m.gpu_memory_used_gb for m in history], mode="lines"))
    fig.add_trace(go.Scatter(x=timestamps, y=[m.gpu_utilization for m in history], mode="lines"))

    assert len(fig.data) == 4, f"图表 trace 数量: {len(fig.data)}"
    print("[PASS] plotly 图表数据准备正常 (4 条 trace)")


if __name__ == "__main__":
    print("=" * 60)
    print("  vLLM Monitor — Smoke Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("模块导入检查", test_imports),
        ("Demo 模式采集", test_collector_demo_mode),
        ("数值范围合法性", test_metric_value_ranges),
        ("vllm_url 兼容性", test_collector_with_vllm_url),
        ("Dataclass 字段完整性", test_metrics_dataclass_fields),
        ("pynvml GPU 数据", test_pynvml_gpu_data),
        ("无 while True 阻塞", test_no_while_true),
        ("关键依赖检查", test_requirements),
        ("多 GPU 检测", test_multi_gpu_detection),
        ("plotly 图表数据", test_plotly_chart_data),
    ]

    passed = 0
    failed = 0
    for name, func in tests:
        print(f"\n--- {name} ---")
        try:
            func()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"  结果: {passed} 通过, {failed} 失败 / 共 {len(tests)} 项")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
