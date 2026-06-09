# vLLM Monitor — 数据记录说明

本文件介绍如何开启数据记录、数据格式说明，以及如何用 CSV / JSONL 文件做后续分析。

---

## 开启数据记录

1. 启动仪表盘：`streamlit run examples/demo_streamlit.py -- --vllm-url http://localhost:8000`
2. 在左侧边栏勾选 **"启用数据记录"**
3. 选择格式：**CSV**（Excel 友好）或 **JSONL**（程序分析友好）
4. 每次采样会自动追加到文件

---

## 文件位置

```
data/metrics/
├── vllm_monitor_20260609_113000.csv     # CSV 格式
├── vllm_monitor_20260609_114500.jsonl    # JSONL 格式
└── .gitkeep
```

- 文件名格式：`vllm_monitor_YYYYMMDD_HHMMSS.csv`
- 每次会话（启动仪表盘）创建一个新文件
- 文件自动保存在 `data/metrics/` 目录下

---

## CSV 字段说明

| 字段 | 示例值 | 来源 | 说明 |
|------|--------|------|------|
| `timestamp` | 1780976000.0 | Python time.time() | Unix 时间戳（秒） |
| `datetime_iso` | 2026-06-09T11:30:00 | datetime.now().isoformat() | 人类可读时间 |
| `gpu_index` | 0 | pynvml | GPU 索引号，从 0 开始 |
| `gpu_name` | NVIDIA GeForce RTX 3060 | pynvml | GPU 型号 |
| `gpu_temperature_c` | 42.0 | pynvml | GPU 核心温度 (°C) |
| `gpu_power_w` | 15.2 | pynvml | GPU 实时功耗 (W) |
| `gpu_memory_used_mb` | 2048.0 | pynvml | GPU 显存已用量 (MB) |
| `gpu_memory_total_mb` | 12288.0 | pynvml | GPU 显存总量 (MB) |
| `gpu_memory_util_percent` | 16.7 | 计算值 | 显存使用率 (%) |
| `gpu_util_percent` | 5 | pynvml | GPU 计算单元利用率 (%) |
| `vllm_url` | http://localhost:8000 | 用户配置 | vLLM 服务地址 |
| `request_count` | 5 | 采集器内存计数 | 累计请求数 |
| `latency_ms` | 200.0 | 采集器内存计数 | 平均延迟 (ms) |
| `throughput_tokens_per_sec` | N/A | — | ⚠️ 尚未接入 |
| `prompt_tokens` | 52 | vLLM /metrics | 累计 prompt token 数 |
| `completion_tokens` | 100 | vLLM /metrics | 累计 generation token 数 |
| `total_tokens` | 152 | 计算值 | prompt + completion tokens |
| `vllm_queue_length` | 0 | vLLM /metrics | 等待处理的请求数 |
| `vllm_running` | 1 | vLLM /metrics | 正在运行的请求数 |
| `vllm_cache_hit_rate` | 0.95 | vLLM /metrics | KV Cache 命中率 |
| `error_message` | N/A | 采集过程 | 异常时为错误描述 |

**N/A** = 该字段当前不可用（尚未接入对应数据源）。

---

## 如何用数据做分析

### 1. GPU 是否繁忙

看 `gpu_util_percent` 列：
- 持续 < 20% → GPU 空闲
- 持续 > 80% → GPU 繁忙

```python
import pandas as pd
df = pd.read_csv("data/metrics/vllm_monitor_20260609_113000.csv")
df["gpu_util_percent"].describe()
```

### 2. 显存是否接近上限

看 `gpu_memory_util_percent` 列：
- 持续 > 90% → 显存瓶颈

### 3. 温度和功耗是否异常

看 `gpu_temperature_c` 和 `gpu_power_w`：
- 温度 > 85°C 需注意
- 功耗长时间接近 TDP 值属正常

### 4. 请求延迟是否偏高

看 `latency_ms`：
- 平均值稳定 → 系统健康
- 突然升高 → 可能触发 GC 或达到并发上限

### 5. 吞吐是否下降

看 `request_count` 的增长斜率：
- 每单位时间增量减少 → 吞吐下降

---

## 当前数据来源状态

| 数据类型 | 状态 | 说明 |
|---------|------|------|
| GPU 利用率 | ✅ 真实 | 来自 pynvml |
| GPU 温度 | ✅ 真实 | 来自 pynvml |
| GPU 功耗 | ✅ 真实 | 来自 pynvml |
| GPU 显存 | ✅ 真实 | 来自 pynvml |
| vLLM 队列长度 | ✅ 真实 | 来自 vLLM /metrics |
| vLLM Token 计数 | ✅ 真实（累计值） | 来自 vLLM /metrics |
| 请求计数 | ⚠️ 内存计数 | 非 vLLM 服务端指标 |
| 请求延迟 | ⚠️ 内存计数 | 非 vLLM 服务端指标 |
| Token 级吞吐 | ❌ 未接入 | 待开发 |
| 服务端 RPS | ❌ 未接入 | 待开发 |
| 缓存命中率 | ⚠️ 部分接入 | vLLM 0.22.1 字段名可能已变更 |
