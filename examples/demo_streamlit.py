"""
vLLM Monitor Demo - Streamlit版本
"""
import argparse
import sys
import time
from vllm_monitor.collector import VLLMMetricsCollector
from vllm_monitor.dashboard import VLLMDashboard


def main():
    parser = argparse.ArgumentParser(description="vLLM Monitor Dashboard")
    parser.add_argument(
        "--vllm-url",
        default=None,
        help="vLLM 服务地址（如 http://localhost:8000），不指定则使用模拟数据",
    )
    # Streamlit 会吃掉自己的参数，我们需要从 sys.argv 中过滤
    args, _ = parser.parse_known_args()

    print("🚀 启动vLLM监控面板...")
    print("📍 访问 http://localhost:8501 查看仪表盘")
    if args.vllm_url:
        print(f"🔗 连接至 vLLM 实例: {args.vllm_url}")
    else:
        print("📊 使用模拟数据（加上 --vllm-url 可连接真实 vLLM 实例）")

    # 创建采集器（demo 模式 / 真实模式）
    collector = VLLMMetricsCollector(vllm_url=args.vllm_url)

    # 模拟一些请求（仅 demo 模式）
    if not args.vllm_url:
        for i in range(5):
            time.sleep(0.5)
            collector.record_request(0.1 + i * 0.05)
            collector.record_communication(1024 * 1024)  # 1MB

    # 启动仪表盘
    dashboard = VLLMDashboard(collector)
    dashboard.run()


if __name__ == "__main__":
    main()
