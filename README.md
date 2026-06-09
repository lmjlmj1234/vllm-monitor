# vLLM Monitor — vLLM 推理性能实时监控面板

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![vLLM](https://img.shields.io/badge/vLLM-0.22+-orange.svg)](https://github.com/vllm-project/vllm)

**实时 GPU + vLLM 推理监控仪表盘**，基于 Streamlit + pynvml。

- ✅ **真实 GPU 数据** — 温度、功耗、显存、利用率（来自 pynvml）
- ✅ **vLLM 指标** — 等待队列、Token 计数、PagedAttention 命中率（来自 /metrics）
- ✅ **数据记录** — 支持 CSV / JSONL 格式保存，便于后续分析
- ✅ **时序图表** — 实时折线图展示 RPS、延迟、显存、利用率趋势
- ✅ **多 GPU 支持** — 自动检测所有 NVIDIA GPU

## 快速开始

### 安装

```bash
git clone https://github.com/YOUR_USERNAME/vllm-monitor.git
cd vllm-monitor
pip install -r requirements.txt
```

### 启动 vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/your/model \
  --port 8000 \
  --trust-remote-code
```

> 也可用 Docker 或其它方式启动，确保 `/metrics` 端点可访问即可。

### 启动监控仪表盘

**Demo 模式**（不需要 vLLM 实例，使用模拟数据）：

```bash
streamlit run examples/demo_streamlit.py
```

**连接真实 vLLM 实例**：

```bash
streamlit run examples/demo_streamlit.py -- --vllm-url http://localhost:8000
```

访问 `http://localhost:8501` 查看仪表盘。

## 数据记录

仪表盘左侧边栏提供 **"启用数据记录"** 开关，可选择 CSV 或 JSONL 格式。

- 文件保存在 `data/metrics/` 目录
- 文件名格式：`vllm_monitor_YYYYMMDD_HHMMSS.csv`
- 每行记录一次采样的完整数据

详见 [docs/data_logging.md](docs/data_logging.md)。

## 数据来源说明

| 指标 | 来源 | 状态 |
|------|------|------|
| GPU 利用率 / 温度 / 功耗 / 显存 | pynvml (NVML) | ✅ 真实数据 |
| vLLM 等待队列 | vLLM /metrics | ✅ 已接入 |
| vLLM Token 计数 | vLLM /metrics | ✅ 已接入 |
| PagedAttention 命中率 | vLLM /metrics | ⚠️ 部分接入 |
| 请求计数 / 延迟 | 内存计数 | ⚠️ Demo 级别 |
| 服务端实际 RPS / Token 吞吐 | — | ❌ 待开发 |

详见 [docs/metrics_guide.md](docs/metrics_guide.md)。

## 项目结构

```
vllm-monitor/
├── vllm_monitor/
│   ├── __init__.py         # 包入口
│   ├── collector.py        # GPU + vLLM 指标采集器
│   ├── dashboard.py        # Streamlit 仪表盘
│   └── recorder.py         # CSV / JSONL 数据记录
├── tests/
│   ├── __init__.py
│   └── test_smoke.py       # 冒烟测试套件（10 项）
├── docs/
│   ├── metrics_guide.md    # 指标含义与解读
│   └── data_logging.md     # 数据记录说明
├── examples/
│   └── demo_streamlit.py   # 启动入口
├── data/
│   └── metrics/            # 监控数据保存目录（被 .gitignore 忽略）
├── requirements.txt        # 依赖清单
├── setup.py                # 包安装配置
└── README.md
```

## 运行测试

轻量冒烟测试（不启动 Streamlit、不加载模型）：

```bash
python tests/test_smoke.py
```

测试内容：模块导入、GPU 数据采集、数值范围合法性、vLLM URL 兼容性、Dataclass 完整性、plolty 图表数据、无死循环检查。

## 依赖

- streamlit — Web 仪表盘
- pynvml — NVIDIA GPU 实时监控
- plotly — 时序图表
- psutil — 系统监控（备用）
- requests — vLLM /metrics HTTP 请求
- （vLLM 可选 — 仅连接真实实例时需要）

## License

MIT
