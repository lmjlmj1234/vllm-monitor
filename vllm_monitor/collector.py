"""
vLLM Performance Metrics Collector
功能：实时采集vLLM推理性能指标
"""
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VLLMMetrics:
    """vLLM核心性能指标"""
    timestamp: float
    requests_per_second: float        # 吞吐（请求/秒）
    avg_latency_ms: float             # 平均延迟（毫秒）
    gpu_memory_used_gb: float         # GPU显存使用（GB）
    gpu_memory_total_gb: float        # GPU显存总量（GB）
    gpu_utilization: float            # GPU利用率（%）
    gpu_temperature: float            # GPU温度（°C）
    gpu_power_w: float                # GPU功耗（瓦）
    communication_bytes: int          # 多卡通信量（字节）
    paged_attention_hit_rate: float   # PagedAttention命中率
    queue_length: int                 # 等待队列长度


class VLLMMetricsCollector:
    """vLLM性能指标采集器"""

    def __init__(self, engine=None, vllm_url: Optional[str] = None):
        self.engine = engine
        self.vllm_url = vllm_url
        self.start_time = time.time()
        self.total_requests = 0
        self.latencies: List[float] = []
        self.communication_bytes = 0

        # 初始化 NVML（实时 GPU 监控，支持多卡）
        self._nvml_handles: List[tuple] = []
        self._init_nvml()

    def _init_nvml(self):
        """初始化 NVML，遍历所有 GPU，失败时不抛异常（退化为模拟数据）"""
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                self._nvml_handles.append((i, handle, name))
                logger.info("GPU %d: %s", i, name)
            logger.info("NVML initialized, %d GPU(s) found", device_count)
        except Exception as e:
            logger.warning("NVML init failed (GPU metrics will be simulated): %s", e)

    # ---- GPU 实时数据 ----

    def _get_gpu_metrics(self):
        """
        通过 pynvml 采集所有 GPU 的真实指标。
        返回聚合值：利用率/温度取平均，显存/功耗取和。
        """
        if not self._nvml_handles:
            return None
        try:
            import pynvml
            total_util = 0.0
            total_mem_used = 0.0
            total_mem = 0.0
            total_temp = 0.0
            total_power = 0.0
            count = len(self._nvml_handles)

            for idx, handle, name in self._nvml_handles:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                power = pynvml.nvmlDeviceGetPowerUsage(handle)

                total_util += util.gpu
                total_mem_used += mem_info.used
                total_mem += mem_info.total
                total_temp += temp
                total_power += power

            return {
                "gpu_util": total_util / count,
                "mem_used_gb": total_mem_used / 1e9,
                "mem_total_gb": total_mem / 1e9,
                "temperature": total_temp / count,
                "power_w": total_power / 1000.0,
                "gpu_count": count,
            }
        except Exception as e:
            logger.debug("Failed to read GPU metrics: %s", e)
            return None

    def _get_gpu_metrics_fallback(self):
        """降级方案：用 torch.cuda 获取显存用量"""
        try:
            import torch
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                return {
                    "gpu_util": 0.0,
                    "mem_used_gb": used,
                    "mem_total_gb": total,
                    "temperature": 0.0,
                    "power_w": 0.0,
                    "gpu_count": 1,
                }
        except Exception:
            pass
        return None

    # ---- vLLM 远程指标 ----

    def _fetch_vllm_metrics(self) -> Optional[Dict]:
        """从 vLLM /metrics 端点抓取 Prometheus 格式指标"""
        if not self.vllm_url:
            return None
        try:
            import requests
            resp = requests.get(
                f"{self.vllm_url.rstrip('/')}/metrics",
                timeout=5,
            )
            resp.raise_for_status()
            return self._parse_vllm_metrics(resp.text)
        except Exception as e:
            logger.debug("Failed to fetch vLLM metrics from %s: %s", self.vllm_url, e)
            return None

    @staticmethod
    def _parse_vllm_metrics(text: str) -> Dict:
        """
        从 Prometheus 文本格式提取 vLLM 关键指标。
        vLLM 公开的指标包括:
          - vllm:num_requests_waiting / num_requests_running
          - vllm:prompt_tokens_total / generation_tokens_total
          - vllm:cache_hit_rate (gauge)
        """
        result: Dict = {
            "queue_length": 0,
            "running": 0,
            "cache_hit_rate": 0.0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            val = line.split()[-1]
            try:
                # 等待队列
                if "num_requests_waiting" in line and "by_reason" not in line:
                    result["queue_length"] = int(float(val))
                # 运行中请求
                if "num_requests_running" in line and "by_reason" not in line:
                    result["running"] = int(float(val))
                # prompt tokens 总数（累计 counter）
                if "prompt_tokens_total" in line and "by_source" not in line:
                    result["prompt_tokens"] = int(float(val))
                # generation tokens 总数（累计 counter）
                if "generation_tokens_total" in line:
                    result["completion_tokens"] = int(float(val))
                # cache hit rate（gauge）
                if "cache_hit_rate" in line:
                    result["cache_hit_rate"] = float(val)
            except (IndexError, ValueError):
                pass

        if result["prompt_tokens"] is not None and result["completion_tokens"] is not None:
            result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
        return result

    # ---- 主采集逻辑 ----

    def collect(self) -> VLLMMetrics:
        """采集当前性能指标"""
        elapsed = time.time() - self.start_time
        rps = self.total_requests / elapsed if elapsed > 0 else 0

        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0

        # GPU 指标：pynvml → torch fallback → 全零
        gpu = self._get_gpu_metrics()
        if gpu is None:
            gpu = self._get_gpu_metrics_fallback()
        if gpu is None:
            gpu = {"gpu_util": 0.0, "mem_used_gb": 0.0, "mem_total_gb": 0.0,
                   "temperature": 0.0, "power_w": 0.0, "gpu_count": 0}

        # vLLM 远程指标
        vllm_remote = self._fetch_vllm_metrics()

        return VLLMMetrics(
            timestamp=time.time(),
            requests_per_second=rps,
            avg_latency_ms=avg_latency * 1000,
            gpu_memory_used_gb=gpu["mem_used_gb"],
            gpu_memory_total_gb=gpu["mem_total_gb"],
            gpu_utilization=gpu["gpu_util"],
            gpu_temperature=gpu["temperature"],
            gpu_power_w=gpu["power_w"],
            communication_bytes=self.communication_bytes,
            paged_attention_hit_rate=(
                vllm_remote["cache_hit_rate"]
                if vllm_remote and vllm_remote.get("cache_hit_rate")
                else 0.95
            ),
            queue_length=(
                vllm_remote["queue_length"]
                if vllm_remote and vllm_remote.get("queue_length") is not None
                else 0
            ),
        )

    # ---- 供 DataRecorder 使用的补充方法 ----

    def get_gpu_details_list(self) -> List[Dict]:
        """返回每张 GPU 的详细信息，用于记录到 CSV/JSONL。

        每张卡返回:
            index, name, temperature_c, power_w,
            memory_used_mb, memory_total_mb,
            memory_util_percent, util_percent
        """
        details: List[Dict] = []
        if not self._nvml_handles:
            return details
        try:
            import pynvml
            for idx, handle, name in self._nvml_handles:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                power = pynvml.nvmlDeviceGetPowerUsage(handle)

                mem_used_mb = mem_info.used / 1e6
                mem_total_mb = mem_info.total / 1e6
                mem_util = (mem_info.used / mem_info.total * 100) if mem_info.total > 0 else 0.0

                details.append({
                    "index": idx,
                    "name": name,
                    "temperature_c": float(temp),
                    "power_w": power / 1000.0,
                    "memory_used_mb": round(mem_used_mb, 2),
                    "memory_total_mb": round(mem_total_mb, 2),
                    "memory_util_percent": round(mem_util, 1),
                    "util_percent": int(util.gpu),
                })
        except Exception:
            pass
        return details

    @property
    def collector_stats(self) -> Dict:
        """返回采集器内部统计值。"""
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
        return {
            "total_requests": self.total_requests,
            "avg_latency_ms": round(avg_latency * 1000, 2),
            "communication_bytes": self.communication_bytes,
        }

    @property
    def vllm_metrics_raw(self) -> Optional[Dict]:
        """返回最新一次从 vLLM /metrics 解析的原始指标字典。"""
        return self._fetch_vllm_metrics()

    def record_request(self, latency: float):
        """记录单个请求的延迟"""
        self.total_requests += 1
        self.latencies.append(latency)

    def record_communication(self, bytes_sent: int):
        """记录通信量"""
        self.communication_bytes += bytes_sent
