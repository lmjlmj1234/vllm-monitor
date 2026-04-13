"""
vLLM Monitor Demo - Streamlit版本
"""
import time
from vllm_monitor.collector import VLLMMetricsCollector, VLLMMetrics
from vllm_monitor.dashboard import VLLMDashboard

def main():
    """运行演示"""
    print("🚀 启动vLLM监控面板...")
    print("📍 访问 http://localhost:8501 查看仪表盘")
    
    # 创建模拟采集器
    collector = VLLMMetricsCollector()
    
    # 模拟一些请求
    for i in range(5):
        time.sleep(0.5)
        collector.record_request(0.1 + i * 0.05)
        collector.record_communication(1024 * 1024)  # 1MB
    
    # 启动仪表盘
    dashboard = VLLMDashboard(collector)
    dashboard.run()

if __name__ == "__main__":
    main()