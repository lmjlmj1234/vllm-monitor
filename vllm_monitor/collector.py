"""
vLLM Performance Metrics Collector
功能：实时采集vLLM推理性能指标
"""
import time
import psutil
import torch
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class VLLMMetrics:
    """vLLM核心性能指标"""
    timestamp: float
    requests_per_second: float      # 吞吐（请求/秒）
    avg_latency_ms: float           # 平均延迟（毫秒）
    gpu_memory_used_gb: float       # GPU显存使用（GB）
    gpu_utilization: float          # GPU利用率（%）
    communication_bytes: int        # 多卡通信量（字节）
    paged_attention_hit_rate: float # PagedAttention命中率
    queue_length: int               # 等待队列长度

class VLLMMetricsCollector:
    """vLLM性能指标采集器"""
    
    def __init__(self, engine=None):
        self.engine = engine
        self.start_time = time.time()
        self.total_requests = 0
        self.latencies = []
        self.communication_bytes = 0
    
    def collect(self) -> VLLMMetrics:
        """采集当前性能指标"""
        # 计算吞吐
        elapsed = time.time() - self.start_time
        rps = self.total_requests / elapsed if elapsed > 0 else 0
        
        # 计算平均延迟
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        
        # 获取GPU显存使用
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        # 获取GPU利用率
        gpu_util = psutil.cpu_percent()  # 简化版，实际应使用nvidia-smi
        
        return VLLMMetrics(
            timestamp=time.time(),
            requests_per_second=rps,
            avg_latency_ms=avg_latency * 1000,
            gpu_memory_used_gb=gpu_mem,
            gpu_utilization=gpu_util,
            communication_bytes=self.communication_bytes,
            paged_attention_hit_rate=0.95,  # 示例值
            queue_length=0  # 示例值
        )
    
    def record_request(self, latency: float):
        """记录单个请求的延迟"""
        self.total_requests += 1
        self.latencies.append(latency)
    
    def record_communication(self, bytes_sent: int):
        """记录通信量"""
        self.communication_bytes += bytes_sent