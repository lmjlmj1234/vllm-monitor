"""
vLLM Monitor — Data Recorder (CSV / JSONL)

记录实时监控数据到本地文件，便于后续分析和学习。
"""

import csv
import json
import os
import logging
from datetime import datetime
from typing import Optional, Literal

logger = logging.getLogger(__name__)

# 输出字段列表（同时也是 CSV 表头 / JSONL 键名）
OUTPUT_FIELDS = [
    # 采样时间
    "timestamp",
    "datetime_iso",
    # GPU 基本信息
    "gpu_index",
    "gpu_name",
    # GPU 实时指标（均来自真实 pynvml）
    "gpu_temperature_c",
    "gpu_power_w",
    "gpu_memory_used_mb",
    "gpu_memory_total_mb",
    "gpu_memory_util_percent",
    "gpu_util_percent",
    # vLLM 连接信息
    "vllm_url",
    # vLLM 推理指标（来自 vLLM /metrics 端点）
    "request_count",
    "latency_ms",
    "throughput_tokens_per_sec",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    # 采集状态
    "error_message",
]

CSV_NULL = "N/A"
JSONL_NULL = None


class DataRecorder:
    """轻量监控数据记录器，支持 CSV 和 JSONL 两种格式。

    用法:
        recorder = DataRecorder(format="csv")
        recorder.record(gpu_data={
            "index": 0, "name": "NVIDIA GeForce RTX 3060",
            "temperature_c": 40.0, "power_w": 10.5,
            "memory_used_mb": 2048, "memory_total_mb": 12288,
            "util_percent": 5,
        }, vllm_url="http://localhost:8000", ...)
        recorder.close()
    """

    def __init__(
        self,
        output_dir: str = "data/metrics",
        fmt: Literal["csv", "jsonl"] = "csv",
    ):
        self.output_dir = os.path.abspath(output_dir)
        self.fmt = fmt
        self._file = None
        self._writer = None
        self._row_count = 0
        self._last_timestamp: Optional[str] = None

        # 自动创建目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 生成文件名：vllm_monitor_YYYYMMDD_HHMMSS.csv / .jsonl
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "csv" if fmt == "csv" else "jsonl"
        self.filepath = os.path.join(self.output_dir, f"vllm_monitor_{ts}.{ext}")

        self._open_file()
        logger.info("DataRecorder initialized: %s", self.filepath)

    # ---- 文件管理 ----

    def _open_file(self):
        """打开文件并写入表头（CSV）或准备写入（JSONL）。"""
        if self.fmt == "csv":
            self._file = open(self.filepath, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
            self._writer.writeheader()
            self._file.flush()
        else:
            self._file = open(self.filepath, "a", encoding="utf-8")
            self._writer = None  # 每行单独 json.dump

    def close(self):
        """关闭文件句柄。"""
        if self._file and not self._file.closed:
            self._file.close()
            logger.info("DataRecorder closed: %s (%d rows)", self.filepath, self._row_count)

    # ---- 核心记录方法 ----

    def record(
        self,
        *,
        gpu_data: dict,
        vllm_url: Optional[str] = None,
        vllm_metrics: Optional[dict] = None,
        collector_stats: Optional[dict] = None,
        error_message: Optional[str] = None,
    ):
        """记录一次采样数据。

        参数:
            gpu_data: pynvml 单卡数据，包含 index, name, temperature_c, power_w,
                      memory_used_mb, memory_total_mb, util_percent
            vllm_url: vLLM 服务地址（字符串）
            vllm_metrics: 从 vLLM /metrics 解析的指标
            collector_stats: 采集器内部统计（total_requests, avg_latency 等）
            error_message: 本次采集的错误信息（None 表示正常）
        """
        now = datetime.now()
        row = {
            "timestamp": now.timestamp(),
            "datetime_iso": now.isoformat(),
            # GPU 数据（必填）
            "gpu_index": gpu_data.get("index", CSV_NULL),
            "gpu_name": gpu_data.get("name", CSV_NULL),
            "gpu_temperature_c": gpu_data.get("temperature_c", CSV_NULL),
            "gpu_power_w": gpu_data.get("power_w", CSV_NULL),
            "gpu_memory_used_mb": gpu_data.get("memory_used_mb", CSV_NULL),
            "gpu_memory_total_mb": gpu_data.get("memory_total_mb", CSV_NULL),
            "gpu_memory_util_percent": gpu_data.get("memory_util_percent", CSV_NULL),
            "gpu_util_percent": gpu_data.get("util_percent", CSV_NULL),
            # vLLM 连接
            "vllm_url": vllm_url if vllm_url else CSV_NULL,
            # vLLM 推理指标
            "request_count": CSV_NULL,
            "latency_ms": CSV_NULL,
            "throughput_tokens_per_sec": CSV_NULL,
            "prompt_tokens": CSV_NULL,
            "completion_tokens": CSV_NULL,
            "total_tokens": CSV_NULL,
            # 错误信息
            "error_message": error_message if error_message else CSV_NULL,
        }

        # 填充 vLLM 指标（非伪造，只有从 /metrics 真正获取到的才写入）
        if vllm_metrics:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                val = vllm_metrics.get(key)
                if val is not None:
                    row[key] = val

        # 填充采集器统计（仅 demo 模式有效，连接 vLLM 时为内存计数）
        if collector_stats:
            if "total_requests" in collector_stats:
                row["request_count"] = collector_stats["total_requests"]
            if "avg_latency_ms" in collector_stats:
                row["latency_ms"] = collector_stats["avg_latency_ms"]

        # 写入
        if self.fmt == "csv":
            self._writer.writerow(row)
        else:
            self._file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

        self._file.flush()
        self._row_count += 1
        self._last_timestamp = row["datetime_iso"]

    # ---- 状态查询 ----

    @property
    def row_count(self) -> int:
        """当前文件已记录的行数。"""
        return self._row_count

    @property
    def last_timestamp(self) -> Optional[str]:
        """最近一次采样的 ISO 时间字符串。"""
        return self._last_timestamp

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
