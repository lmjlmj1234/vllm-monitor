"""
vLLM实时监控仪表盘（Streamlit版）
"""
import time
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vllm_monitor.collector import VLLMMetricsCollector
from vllm_monitor.recorder import DataRecorder


# ---- 指标说明（下拉菜单显示用） ----
METRICS_HELP = {
    "GPU 利用率 (%)": (
        "来自 pynvml：nvmlDeviceGetUtilizationRates().gpu\n\n"
        "表示 GPU 计算单元在一段时间内的繁忙比例。\n"
        "- 0-20%：基本空闲\n"
        "- 20-60%：轻度负载\n"
        "- 60-90%：繁忙\n"
        "- 90-100%：接近满载\n\n"
        "注意：这是「显存接口」利用率还是「计算单元」利用率？这个是计算单元利用率。"
    ),
    "GPU 温度 (°C)": (
        "来自 pynvml：nvmlDeviceGetTemperature()\n\n"
        "GPU 核心温度。\n"
        "- < 60°C：正常\n"
        "- 60-80°C：偏高，注意散热\n"
        "- 80-85°C：警戒线\n"
        "- > 85°C：过热风险，可能降频\n\n"
        "RTX 3060 典型满载温度在 65-75°C。"
    ),
    "GPU 功耗 (W)": (
        "来自 pynvml：nvmlDeviceGetPowerUsage()\n\n"
        "GPU 实时功耗。RTX 3060 的 TDP 为 170W。\n"
        "- < 30W：空闲\n"
        "- 30-80W：轻度负载\n"
        "- 80-150W：中高负载\n"
        "- > 150W：接近满载\n\n"
        "结合利用率和温度一起看，可以判断散热是否正常。"
    ),
    "GPU 显存 (GB)": (
        "来自 pynvml：nvmlDeviceGetMemoryInfo()\n\n"
        "GPU 显存使用量 / 总量。\n"
        "- 已用显存持续接近总量 → 显存瓶颈\n"
        "- 对于 LLM 推理，模型权重 + KV Cache 是主要占用\n\n"
        "Qwen2.5-0.5B 约占用 0.9-1.5 GB 显存。"
    ),
    "吞吐 (RPS)": (
        "= total_requests / 运行时间（秒）\n\n"
        "每秒处理的请求数。\n"
        "当前在 Demo 模式下通过模拟请求产生；\n"
        "连接 vLLM 后此项为采集器内存计数（非 vLLM 服务端指标）。\n\n"
        "⚠️ vLLM 服务端的真实 RPS 尚未接入。"
    ),
    "平均延迟 (ms)": (
        "= 累计请求延迟 / 总请求数\n\n"
        "请求的平均处理延迟。\n"
        "当前在 Demo 模式下通过 record_request() 模拟；\n"
        "连接 vLLM 后此项为采集器内存计数。\n\n"
        "⚠️ vLLM 服务端的真实延迟尚未接入。"
    ),
    "等待队列": (
        "来自 vLLM /metrics：vllm:num_requests_waiting\n\n"
        "等待处理的请求数量。\n"
        "如果队列持续 > 0，说明服务端处理速度跟不上请求到达速度。"
    ),
    "PagedAttention 命中率": (
        "来自 vLLM /metrics：vllm:cache_hit_rate\n\n"
        "KV Cache 命中率。\n"
        "- 命中率高 → 减少重复计算，提高推理速度\n"
        "- 命中率低 → 新序列多，预处理量大\n\n"
        "⚠️ vLLM 0.22.1 中的 cache_hit_rate 指标名可能已变更，"
        "当前为默认值 0.95。"
    ),
}

REAL_VLLM_FIELDS = {
    "等待队列": True,       # 已接入 vllm:num_requests_waiting
    "PagedAttention 命中率": "部分",  # 命名可能变化
}

REAL_GPU_FIELDS = {
    "GPU 利用率": True,
    "GPU 温度": True,
    "GPU 功耗": True,
    "GPU 显存": True,
}


class VLLMDashboard:
    """vLLM监控仪表盘"""

    def __init__(self, collector: VLLMMetricsCollector):
        self.collector = collector
        self.max_history = 100

    def run(self):
        """启动仪表盘"""
        st.set_page_config(
            page_title="vLLM Monitor",
            page_icon="🚀",
            layout="wide"
        )

        st.title("🚀 vLLM 实时性能监控")
        st.markdown("基于 [vLLM](https://github.com/vllm-project/vllm) 的推理性能监控面板")

        # ---- 侧边栏 ----
        with st.sidebar:
            st.header("📊 配置")
            refresh_interval = st.slider("刷新间隔（秒）", 0.5, 5.0, 1.0)

            st.markdown("---")

            # ---- 数据记录控制 ----
            st.header("💾 数据记录")
            enable_logging = st.checkbox("启用数据记录", value=False,
                                         help="开启后将实时监控数据保存到本地文件")

            log_format = "csv"
            recorder = None
            if enable_logging:
                log_format = st.radio("保存格式", ["csv", "jsonl"],
                                      index=0, horizontal=True,
                                      help="CSV 适合 Excel 打开，JSONL 适合程序分析")
                # 初始化 recorder 并保存到 session_state
                if "recorder" not in st.session_state or st.session_state.get("log_format") != log_format:
                    st.session_state.recorder = DataRecorder(fmt=log_format)
                    st.session_state.log_format = log_format
                recorder = st.session_state.recorder

                st.markdown(f"**日志文件**  \n`{recorder.filepath}`")
                st.caption(f"已记录 {recorder.row_count} 条 | "
                           f"最近采样: {recorder.last_timestamp or 'N/A'}")

            st.markdown("---")
            st.info("💡 调整刷新间隔可平衡实时性和性能开销")

            # ---- 指标说明 ----
            with st.expander("📖 指标说明", expanded=False):
                for name, desc in METRICS_HELP.items():
                    st.markdown(f"**{name}**")
                    st.caption(desc.split('\n\n')[0])  # 只显示第一段简短说明
                    st.markdown("---", unsafe_allow_html=True)

            # ---- 数据来源说明 ----
            with st.expander("🔍 数据来源状态", expanded=False):
                st.markdown("**✅ 真实数据 (pynvml)**")
                for f, _ in REAL_GPU_FIELDS.items():
                    st.markdown(f"- {f}")
                st.markdown("")
                st.markdown("**⚠️ 部分接入 (vLLM /metrics)**")
                st.markdown("- 等待队列")
                st.markdown("- PagedAttention 命中率（字段名可能变化）")
                st.markdown("")
                st.markdown("**❌ 尚未接入**")
                st.markdown("- 服务端实际 RPS")
                st.markdown("- 服务端实际延迟")
                st.markdown("- Token 级吞吐")
                st.markdown("- Prompt/Generation tokens 按请求分布")

        # ---- session_state 初始化 ----
        if "metrics_history" not in st.session_state:
            st.session_state.metrics_history = []

        # ---- 采集一次指标 ----
        metrics = self.collector.collect()
        history = st.session_state.metrics_history
        history.append(metrics)
        if len(history) > self.max_history:
            history.pop(0)

        # ---- 数据记录（采集后写入） ----
        if recorder:
            gpu_list = self.collector.get_gpu_details_list()
            vllm_raw = self.collector.vllm_metrics_raw
            collector_stats = self.collector.collector_stats
            for gpu in gpu_list:
                recorder.record(
                    gpu_data=gpu,
                    vllm_url=self.collector.vllm_url,
                    vllm_metrics=vllm_raw,
                    collector_stats=collector_stats,
                )

        # ---- 顶部指标卡片 ----
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📈 吞吐 (RPS)", f"{metrics.requests_per_second:.2f}")
        col2.metric("⏱️ 平均延迟 (ms)", f"{metrics.avg_latency_ms:.2f}")
        col3.metric("💾 GPU显存 (GB)", f"{metrics.gpu_memory_used_gb:.2f} / {metrics.gpu_memory_total_gb:.2f}")
        col4.metric("🔄 等待队列", metrics.queue_length)

        st.markdown("---")

        # ---- 实时指标详情 ----
        st.subheader("📊 实时指标详情")
        col_a, col_b = st.columns(2)

        with col_a:
            st.metric("GPU利用率", f"{metrics.gpu_utilization:.1f}%",
                      help=METRICS_HELP["GPU 利用率 (%)"])
            st.metric("GPU温度", f"{metrics.gpu_temperature:.0f}°C",
                      help=METRICS_HELP["GPU 温度 (°C)"])
            st.metric("GPU功耗", f"{metrics.gpu_power_w:.1f} W",
                      help=METRICS_HELP["GPU 功耗 (W)"])

        with col_b:
            st.metric("PagedAttention命中率", f"{metrics.paged_attention_hit_rate:.2%}",
                      help=METRICS_HELP["PagedAttention 命中率"])
            st.metric("通信量", f"{metrics.communication_bytes / 1e6:.2f} MB",
                      help="累计通信量（record_communication）")
            st.metric("运行时间", f"{time.time() - self.collector.start_time:.1f} s")

        # ---- 数据来源提示 ----
        if not self.collector.vllm_url:
            st.info(
                "📢 Demo 模式 — 使用模拟数据。添加 `--vllm-url` 连接真实 vLLM 实例。"
            )
        else:
            st.caption(
                f"🔗 已连接 vLLM: {self.collector.vllm_url}  |  "
                f"GPU 数据来源: pynvml ✅  |  "
                f"vLLM 推理指标: 部分接入 ⚠️  "
                f"(服务端 RPS / 延迟 / Token 吞吐尚未接入)"
            )

        # ---- 时序图表 ----
        if len(history) >= 2:
            st.markdown("---")
            st.subheader("📈 历史趋势")

            timestamps = [datetime.fromtimestamp(m.timestamp) for m in history]

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("吞吐 (RPS)", "平均延迟 (ms)", "GPU显存 (GB)", "GPU利用率 (%)"),
                vertical_spacing=0.12,
                horizontal_spacing=0.08,
            )

            fig.add_trace(
                go.Scatter(x=timestamps, y=[m.requests_per_second for m in history],
                           mode="lines", name="RPS", line=dict(color="#00cc96", width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=timestamps, y=[m.avg_latency_ms for m in history],
                           mode="lines", name="延迟", line=dict(color="#ab63fa", width=2)),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=timestamps, y=[m.gpu_memory_used_gb for m in history],
                           mode="lines", name="已用", line=dict(color="#ef553b", width=2)),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=timestamps, y=[m.gpu_utilization for m in history],
                           mode="lines", name="GPU%", line=dict(color="#636efa", width=2)),
                row=2, col=2
            )

            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=30, b=20),
                showlegend=False,
                template="plotly_dark",
            )
            fig.update_xaxes(title_text="时间", row=2, col=1)
            fig.update_xaxes(title_text="时间", row=2, col=2)
            fig.update_yaxes(title_text="RPS", row=1, col=1)
            fig.update_yaxes(title_text="ms", row=1, col=2)

            st.plotly_chart(fig, use_container_width=True)

        # 调度下一次 rerun
        time.sleep(refresh_interval)
        st.rerun()
