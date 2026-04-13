"""
vLLM实时监控仪表盘（Streamlit版）
"""
import streamlit as st
import time
from datetime import datetime
from vllm_monitor.collector import VLLMMetricsCollector

class VLLMDashboard:
    """vLLM监控仪表盘"""
    
    def __init__(self, collector: VLLMMetricsCollector):
        self.collector = collector
        self.metrics_history = []
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
        
        # 侧边栏配置
        with st.sidebar:
            st.header("📊 配置")
            refresh_interval = st.slider("刷新间隔（秒）", 0.5, 5.0, 1.0)
            st.markdown("---")
            st.info("💡 提示：调整刷新间隔可平衡实时性和性能开销")
        
        # 主面板指标卡片
        col1, col2, col3, col4 = st.columns(4)
        
        rps_card = col1.metric("📈 吞吐 (RPS)", "0.0")
        latency_card = col2.metric("⏱️ 平均延迟 (ms)", "0.0")
        gpu_mem_card = col3.metric("💾 GPU显存 (GB)", "0.0")
        queue_card = col4.metric("🔄 等待队列", "0")
        
        st.markdown("---")
        
        # 实时更新循环
        placeholder = st.empty()
        
        while True:
            metrics = self.collector.collect()
            self.metrics_history.append(metrics)
            
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)
            
            # 更新指标卡片
            rps_card.metric("📈 吞吐 (RPS)", f"{metrics.requests_per_second:.2f}")
            latency_card.metric("⏱️ 平均延迟 (ms)", f"{metrics.avg_latency_ms:.2f}")
            gpu_mem_card.metric("💾 GPU显存 (GB)", f"{metrics.gpu_memory_used_gb:.2f}")
            queue_card.metric("🔄 等待队列", metrics.queue_length)
            
            # 显示详细信息
            with placeholder.container():
                st.subheader("📊 实时指标详情")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("GPU利用率", f"{metrics.gpu_utilization:.1f}%")
                    st.metric("通信量", f"{metrics.communication_bytes / 1e6:.2f} MB")
                
                with col_b:
                    st.metric("PagedAttention命中率", f"{metrics.paged_attention_hit_rate:.2%}")
                    st.metric("运行时间", f"{time.time() - self.collector.start_time:.1f} s")
            
            time.sleep(refresh_interval)