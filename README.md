# 🚀 vLLM Monitor - vLLM推理性能实时监控面板

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![vLLM](https://img.shields.io/badge/vLLM-0.4.0+-orange.svg)](https://github.com/vllm-project/vllm)

## 📖 项目简介

**vLLM Monitor** 是一个基于 Streamlit 的 vLLM 推理性能实时监控工具，帮助开发者和运维人员：

- ✅ 实时监控吞吐、延迟、显存使用
- ✅ 可视化多卡通信开销（张量并行）
- ✅ 分析 PagedAttention 缓存命中率
- ✅ 导出 Prometheus 指标用于 Grafana

## 🌟 核心功能

| 功能 | 说明 |
|------|------|
| **实时指标采集** | 吞吐、延迟、显存、GPU利用率 |
| **多卡通信监控** | 张量并行通信量统计 |
| **PagedAttention分析** | KV缓存命中率、碎片率 |
| **Streamlit可视化** | 交互式仪表盘 |

## 🚀 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/YOUR_USERNAME/vllm-monitor.git
cd vllm-monitor

# 安装依赖
pip install -r requirements.txt
