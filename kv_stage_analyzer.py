# -*- coding: utf-8 -*-
"""PD 请求五阶段耗时分析（KV Stage Analyzer）。

解析 vllm PD 分离部署的请求级阶段日志，统计每请求五阶段耗时：
    P 等待 / P 执行 / KV 拉取 / D 等待 / D 执行
并将 worker 级 KV 传输明细聚合到请求，提供 KV 等待分解诊断。

数据源（配套 vllm 侧上报《vLLM-Ascend PD分离 KV传输时间Metrics上报开发指南》v2）：
    P 日志: REQ_STAGE_METRIC request_id=<id> p_waiting_s=<f> p_running_s=<f>
    D 日志: REQ_STAGE_METRIC request_id=<id> kv_transfer_s=<f> d_waiting_s=<f> d_running_s=<f>
    D 日志: KV_XFER_METRIC request_id=<id>-<uuid> elapsed_ms=<f> bytes=<int> local_blocks=<int>
                  local_ip=<ip> local_device_id=<int> remote_session_id=<ip:port>
    D 日志: KV_XFER_FAIL ret=-1, continuing        （无 request_id，仅全局计数）

采集方式：offset 增量读取（start 记录文件大小，collect 只读新增段），
天然过滤 warmup 阶段与历史请求，无需时间戳/request_id 白名单。
"""

import csv
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

_REQ_STAGE_RE = re.compile(r"REQ_STAGE_METRIC\s+request_id=(\S+)\s+(.*)")
# vllm 日志时间戳前缀，如 "INFO 10-16 05:25:01 [loggers.py:1231]"（年份缺失，取当前年）
_TS_RE = re.compile(r"(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})")
# vllm 侧打点的毫秒级 epoch 时间戳（REQ_STAGE_METRIC 行内 ts_ms 字段），优先于秒级行前缀
_TS_MS_RE = re.compile(r"\bts_ms=(\d+(?:\.\d+)?)")
_KV_XFER_RE = re.compile(
    r"KV_XFER_METRIC\s+request_id=(\S+)\s+elapsed_ms=(\S+)\s+bytes=(\d+)"
    r"\s+local_blocks=(\d+)\s+local_ip=(\S+)\s+local_device_id=(\d+)"
)
_KV_XFER_FAIL_RE = re.compile(r"KV_XFER_FAIL")
_FIELD_RE = re.compile(r"(\w+)=(\S+)")

# 五阶段：顺序 = 图 11 堆叠自下而上
STAGE_KEYS = ("p_waiting", "p_running", "kv_transfer", "d_waiting", "d_running")
STAGE_LABELS = {
    "p_waiting": "P 等待",
    "p_running": "P 执行",
    "kv_transfer": "KV 拉取",
    "d_waiting": "D 等待",
    "d_running": "D 执行",
}
STAGE_COLORS = {
    "p_waiting": "#94a3b8",
    "p_running": "#3b82f6",
    "kv_transfer": "#f97316",
    "d_waiting": "#cbd5e1",
    "d_running": "#22c55e",
}
# 图 11 显示上限（超过时均匀降采样，统计与 CSV 始终全量）
MAX_PLOT_REQUESTS = 2000


def _percentile(sorted_values: List[float], pct: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def _read_from_offset(path: str, offset: int) -> Tuple[str, int]:
    """从 offset 读到 EOF；文件变小（轮转）→ 全量重读并告警。"""
    size = os.path.getsize(path)
    if size < offset:
        logging.warning(
            "[kv_stage] 日志疑似已轮转：%s 当前 %d 字节 < 记录的 offset %d，回退全文件解析（可能混入历史请求）",
            path, size, offset,
        )
        offset = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        fh.seek(offset)
        text = fh.read()
    return text, size


class KVStageAnalyzer:
    """offset 增量采集 + 解析 + join + 聚合。

    用法：
        analyzer = KVStageAnalyzer.from_config()
        if analyzer.enabled:
            analyzer.start()          # full_dataset 阶段开始前（幂等，多轮只记一次）
            ... 压测 ...
            analyzer.collect()        # 每轮 full_dataset 结束后（增量，自动衔接）
            analyzer.summary()        # 统计
            analyzer.build_bar_traces()  # 图 11 数据
    """

    def __init__(self, log_p: str = "", log_d: str = ""):
        self.log_p = (log_p or "").strip()
        self.log_d = (log_d or "").strip()
        self.enabled = bool(self.log_p) and bool(self.log_d)
        self.records: List[Dict[str, Any]] = []
        self.parse_errors = 0
        self._started = False
        self._read_p = 0
        self._read_d = 0

    # ---------- 初始化 ----------

    @classmethod
    def from_config(cls) -> "KVStageAnalyzer":
        log_p = log_d = ""
        try:
            import config as _config  # noqa: PLC0415

            log_p = getattr(_config, "KV_STAGE_LOG_P", "") or ""
            log_d = getattr(_config, "KV_STAGE_LOG_D", "") or ""
        except Exception:  # noqa: BLE001
            logging.warning("[kv_stage] 读取 config 失败，功能禁用")
        analyzer = cls(log_p, log_d)
        if analyzer.enabled:
            logging.info("[kv_stage] PD 请求五阶段耗时分析已启用: P=%s D=%s", log_p, log_d)
        return analyzer

    def start(self) -> None:
        """记录日志当前 offset（幂等：多次调用只记录首次之后的位置）。"""
        if not self.enabled or self._started:
            return
        missing = []
        for path in (self.log_p, self.log_d):
            if not os.path.exists(path):
                missing.append(path)
        if missing:
            logging.warning(
                "[kv_stage] 日志文件不存在，功能降级禁用（不中断测试）：%s", ", ".join(missing)
            )
            self.enabled = False
            return
        self._read_p = os.path.getsize(self.log_p)
        self._read_d = os.path.getsize(self.log_d)
        self._started = True
        logging.info(
            "[kv_stage] baseline offset: P=%d D=%d（此前日志与 warmup 阶段将被排除）",
            self._read_p, self._read_d,
        )

    # ---------- 解析 ----------

    def _parse_stage_lines(self, text: str) -> List[Dict[str, Any]]:
        rows = []
        year = datetime.now().year
        for line in text.splitlines():
            match = _REQ_STAGE_RE.search(line)
            if not match:
                continue
            request_id, fields_text = match.group(1), match.group(2)
            # 时间戳优先级：行内 ts_ms=<epoch毫秒>（vllm 侧打点，毫秒精度）> 行前缀 MM-DD HH:MM:SS（秒级）
            ts = None
            ts_ms_match = _TS_MS_RE.search(fields_text)
            if ts_ms_match:
                try:
                    ts = datetime.fromtimestamp(float(ts_ms_match.group(1)) / 1000.0)
                except (ValueError, OSError, OverflowError):
                    ts = None
            if ts is None:
                ts_match = _TS_RE.search(line[:match.start() + 40])
                if ts_match:
                    try:
                        ts = datetime(year, int(ts_match.group(1)), int(ts_match.group(2)),
                                      int(ts_match.group(3)), int(ts_match.group(4)), int(ts_match.group(5)))
                    except ValueError:
                        ts = None
            fields = dict(_FIELD_RE.findall(fields_text))
            try:
                row: Dict[str, Any] = {"request_id": request_id, "ts": ts}
                for name, key in (
                    ("p_waiting_s", "p_waiting"),
                    ("p_running_s", "p_running"),
                    ("kv_transfer_s", "kv_transfer"),
                    ("d_waiting_s", "d_waiting"),
                    ("d_running_s", "d_running"),
                ):
                    if name in fields:
                        row[key] = float(fields[name])
                rows.append(row)
            except (TypeError, ValueError):
                self.parse_errors += 1
        return rows

    def _parse_worker_lines(self, text: str) -> Dict[str, Dict[str, Any]]:
        """KV_XFER_METRIC 按 request_id（去 -<uuid> 后缀）聚合。"""
        agg: Dict[str, Dict[str, Any]] = {}
        for line in text.splitlines():
            match = _KV_XFER_RE.search(line)
            if not match:
                continue
            # 完整 id 形如 chatcmpl-<uuid4>-<传输会话后缀>，去掉最后一段即请求 id
            request_id = match.group(1).rsplit("-", 1)[0]
            try:
                elapsed_ms = float(match.group(2))
                bytes_ = int(match.group(3))
                device_id = int(match.group(6))
            except (TypeError, ValueError):
                self.parse_errors += 1
                continue
            item = agg.setdefault(
                request_id,
                {"count": 0, "bytes_total": 0, "elapsed_min": None,
                 "elapsed_max": None, "elapsed_sum": 0.0, "device_ids": []},
            )
            item["count"] += 1
            item["bytes_total"] += bytes_
            item["elapsed_sum"] += elapsed_ms
            item["elapsed_min"] = elapsed_ms if item["elapsed_min"] is None else min(item["elapsed_min"], elapsed_ms)
            item["elapsed_max"] = elapsed_ms if item["elapsed_max"] is None else max(item["elapsed_max"], elapsed_ms)
            item["device_ids"].append(device_id)
        return agg

    @staticmethod
    def _most_common_worker_count(agg: Dict[str, Dict[str, Any]]) -> int:
        if not agg:
            return 0
        counter: Dict[int, int] = {}
        for item in agg.values():
            counter[item["count"]] = counter.get(item["count"], 0) + 1
        return max(counter, key=counter.get)

    # ---------- 采集与 join ----------

    def collect(self, phase: str = "full_dataset") -> List[Dict[str, Any]]:
        """增量读取两份日志并解析 join；返回本轮新增记录（records 持续累积）。

        phase 标记本批记录所属测试阶段（prefix_warmup / full_dataset / full_dataset_round_N），
        汇总统计（summary/csv_extra_metrics）仅计入 full 系阶段。
        """
        if not self.enabled or not self._started:
            return []
        text_p, self._read_p = _read_from_offset(self.log_p, self._read_p)
        text_d, self._read_d = _read_from_offset(self.log_d, self._read_d)

        p_rows = self._parse_stage_lines(text_p)
        d_rows = self._parse_stage_lines(text_d)
        workers = self._parse_worker_lines(text_d)
        fail_total = len(_KV_XFER_FAIL_RE.findall(text_d))
        expected_workers = self._most_common_worker_count(workers)

        p_by_id = {row["request_id"]: row for row in p_rows}
        # D 行 = 请求完成点，以 D 为主序
        known = {rec["request_id"] for rec in self.records}
        new_records = []
        for d_row in d_rows:
            rid = d_row["request_id"]
            if rid in known:
                continue
            p_row = p_by_id.get(rid, {})
            worker = workers.get(rid)
            rec: Dict[str, Any] = {
                "request_id": rid,
                "seq": len(self.records) + len(new_records) + 1,
                "phase": phase,
                "t_p_end": p_row.get("ts"),
                "t_d_end": d_row.get("ts"),
                "p_waiting": p_row.get("p_waiting"),
                "p_running": p_row.get("p_running"),
                "kv_transfer": d_row.get("kv_transfer"),
                "d_waiting": d_row.get("d_waiting"),
                "d_running": d_row.get("d_running"),
                "missing_p": "p_running" not in p_row,
                "worker_count": worker["count"] if worker else 0,
                "worker_failed": max(0, expected_workers - worker["count"]) if worker else expected_workers,
                "worker_device_ids": worker["device_ids"] if worker else [],
                "kv_bytes_total": worker["bytes_total"] if worker else 0,
                "kv_elapsed_ms_max": worker["elapsed_max"] if worker else None,
                "kv_elapsed_ms_avg": (worker["elapsed_sum"] / worker["count"]) if worker else None,
            }
            # KV 等待分解：调度器视角总时间 − 最慢 worker 纯传输
            if rec["kv_transfer"] is not None and rec["kv_elapsed_ms_max"] is not None:
                rec["kv_wait_overhead"] = max(0.0, rec["kv_transfer"] - rec["kv_elapsed_ms_max"] / 1000.0)
            else:
                rec["kv_wait_overhead"] = None
            new_records.append(rec)

        # P 侧有而 D 侧缺失的请求（请求未完成/失败）：仅告警计数，不生成记录
        missing_d = [rid for rid in p_by_id if rid not in {r["request_id"] for r in d_rows}]
        self.records.extend(new_records)
        self._xfer_fail_total = getattr(self, "_xfer_fail_total", 0) + fail_total
        self._expected_workers = expected_workers
        logging.info(
            "[kv_stage] collect: 本轮新增 %d 条（P 行 %d / D 行 %d / worker 明细 %d 个请求，P 缺失 %d / D 缺失 %d）",
            len(new_records), len(p_rows), len(d_rows), len(workers),
            sum(1 for r in new_records if r["missing_p"]), len(missing_d),
        )
        if self.parse_errors:
            logging.warning("[kv_stage] 累计格式异常行 %d 条（已跳过）", self.parse_errors)
        return new_records

    # ---------- 与 aisbench 客户端明细对齐 ----------

    @staticmethod
    def _is_full_phase(phase: str) -> bool:
        return str(phase or "").startswith("full_dataset")

    def align_with_aisbench_details(self, perf_dir: str) -> bool:
        """把 full 系记录与 aisbench 客户端明细（gsm8k_details.jsonl + db_data）对齐。

        对齐原理（aisbench 不记录 vllm response id，无法直接 join）：
        - details.jsonl 每行的 time_points（monotonic 时钟，微秒精度）存在 db_data/*.db
          的 numpy_store，time_points[0] = 客户端发起时间，[-1] = 完成时间（收到最后 token）
        - details 行 id = 数据集顺序 = 客户端发起顺序（即图 1 的 Request Index 语义）
        - 客户端完成顺序（lasts 排序）与服务端 D 侧完成顺序（本工具 records 的 seq 序）
          完全同序（完成间隔 ≫ 网络抖动），按 rank 一一对应
        - 对齐后 rec 获得 client_id（= 图 1 Request Index）与 client_span（交叉校验用）

        多轮 --repeat 时仅对齐最后一个 full 阶段组（details 被最后一轮覆盖）。
        返回是否成功对齐（numpy/db/details 缺失时返回 False，编号退化为完成序）。
        """
        full_records = [r for r in self.records if self._is_full_phase(r.get("phase"))]
        if not full_records or not perf_dir or not os.path.isdir(perf_dir):
            return False
        details_path = os.path.join(perf_dir, "gsm8k_details.jsonl")
        if not os.path.exists(details_path):
            logging.warning("[kv_stage] 未找到 %s，编号退化为服务端完成序", details_path)
            return False
        try:
            import io
            import sqlite3

            import numpy as np
        except ImportError:
            logging.warning("[kv_stage] 缺少 numpy/sqlite3 依赖，编号退化为服务端完成序")
            return False

        db_dir = os.path.join(perf_dir, "db_data")
        db_files = [os.path.join(db_dir, f) for f in os.listdir(db_dir) if f.endswith(".db")] if os.path.isdir(db_dir) else []
        rows = []
        try:
            with open(details_path, "r", encoding="utf-8") as fh:
                detail_lines = [json.loads(line) for line in fh if line.strip()]
            if not detail_lines:
                logging.warning("[kv_stage] details.jsonl 为空，编号退化为服务端完成序")
                return False
            conn_cache = {}
            for row in detail_lines:
                ref = (row.get("time_points") or {}).get("__db_ref__")
                if ref is None:
                    continue
                arr = None
                for db_path in db_files:
                    if db_path not in conn_cache:
                        conn_cache[db_path] = sqlite3.connect(db_path)
                    try:
                        blob = conn_cache[db_path].execute(
                            "SELECT arr_blob FROM numpy_store WHERE id=?", (ref,)).fetchone()
                    except sqlite3.Error:
                        continue
                    if blob:
                        arr = np.load(io.BytesIO(blob[0]))
                        break
                if arr is None or len(arr) < 2:
                    continue
                rows.append({
                    "id": row.get("id"),
                    "first": float(arr[0]),
                    "last": float(arr[-1]),
                    "span": float(arr[-1] - arr[0]),
                })
        except Exception as exc:  # noqa: BLE001
            logging.warning("[kv_stage] 读取 details 失败（%s），编号退化为服务端完成序", exc)
            return False
        if not rows:
            logging.warning("[kv_stage] details 无可用 time_points，编号退化为服务端完成序")
            return False

        # 多轮 --repeat：仅对齐最后一个 full 阶段组（details 只保留最后一轮）
        last_phase = full_records[-1].get("phase")
        align_group = [r for r in full_records if r.get("phase") == last_phase]
        # 客户端完成顺序（lasts 升序）↔ 服务端完成顺序（records 序，即 seq 序）
        rows_sorted = sorted(rows, key=lambda r: r["last"])
        if len(rows_sorted) != len(align_group):
            logging.warning(
                "[kv_stage] details 行数 %d 与最后一个 full 阶段请求数 %d 不一致，编号退化为服务端完成序",
                len(rows_sorted), len(align_group),
            )
            return False
        misaligned = 0
        for rec, cli in zip(align_group, rows_sorted):
            rec["client_id"] = cli["id"]
            rec["client_span"] = cli["span"]
            stages_sum = sum(rec.get(k) or 0.0 for k in STAGE_KEYS)
            # 交叉校验：客户端 E2E 与服务端五阶段之和应接近（差值 ≈ 网络 + proxy 间隙）
            if abs(cli["span"] - stages_sum) > 1.5:
                misaligned += 1
        if misaligned:
            logging.warning("[kv_stage] %d/%d 条对齐校验偏差>1.5s（客户端跨度 vs 五阶段和），可能存在错配", misaligned, len(align_group))
        logging.info(
            "[kv_stage] 已与 aisbench 客户端明细对齐：最后 full 阶段 %d 条 → Request Index（图 1 编号）",
            len(align_group),
        )
        return True

    # ---------- 统计 ----------

    def summary(self) -> Dict[str, Any]:
        all_full = [r for r in self.records if self._is_full_phase(r.get("phase"))]
        records = [r for r in all_full if not r["missing_p"]]
        result: Dict[str, Any] = {
            "count_total": len(all_full),
            "count_joined": len(records),
            "count_warmup": len(self.records) - len(all_full),
            "missing_p": sum(1 for r in all_full if r["missing_p"]),
            "xfer_fail_total": getattr(self, "_xfer_fail_total", 0),
            "expected_workers": getattr(self, "_expected_workers", 0),
            "stages": {},
        }
        total_sums: Dict[str, float] = {}
        for key in STAGE_KEYS:
            values = sorted(r[key] for r in records if r.get(key) is not None)
            if not values:
                continue
            total_sums[key] = sum(values)
            result["stages"][key] = {
                "avg": sum(values) / len(values),
                "p50": _percentile(values, 0.50),
                "p90": _percentile(values, 0.90),
                "max": values[-1],
                "sum": total_sums[key],
                "count": len(values),
            }
        grand = sum(total_sums.values())
        for key, info in result["stages"].items():
            info["share"] = (info["sum"] / grand) if grand > 0 else 0.0
        # KV 等待分解
        overhead = sorted(r["kv_wait_overhead"] for r in records if r.get("kv_wait_overhead") is not None)
        pure = sorted(r["kv_elapsed_ms_max"] / 1000.0 for r in records if r.get("kv_elapsed_ms_max") is not None)
        if overhead:
            result["kv_wait_overhead"] = {
                "avg": sum(overhead) / len(overhead),
                "p90": _percentile(overhead, 0.90),
                "max": overhead[-1],
            }
        if pure:
            result["kv_transfer_pure_max"] = {
                "avg": sum(pure) / len(pure),
                "p90": _percentile(pure, 0.90),
                "max": pure[-1],
            }
        return result

    def log_summary(self) -> None:
        """打屏汇总表（含 KV 等待分解）。"""
        info = self.summary()
        if not info["count_total"]:
            logging.info("[kv_stage] 未采集到请求记录")
            return
        lines = ["=" * 96, f"PD 请求五阶段耗时汇总（{info['count_joined']}/{info['count_total']} 条 P/D join 成功）", "-" * 96]
        header = f"{'阶段':<14}{'avg(s)':>10}{'P50(s)':>10}{'P90(s)':>10}{'max(s)':>10}{'总和占比':>10}"
        lines.append(header)
        for key in STAGE_KEYS:
            stage = info["stages"].get(key)
            if not stage:
                lines.append(f"{STAGE_LABELS[key]:<14}{'-':>10}")
                continue
            lines.append(
                f"{STAGE_LABELS[key]:<14}{stage['avg']:>10.4f}{stage['p50']:>10.4f}"
                f"{stage['p90']:>10.4f}{stage['max']:>10.4f}{stage['share']:>9.1%}"
            )
        overhead = info.get("kv_wait_overhead")
        pure = info.get("kv_transfer_pure_max")
        if overhead and pure:
            lines.append("-" * 96)
            lines.append(
                f"KV 分解:      纯传输(max worker) avg={pure['avg']:.4f}s P90={pure['p90']:.4f}s | "
                f"等待开销(建连+排队+失败超时) avg={overhead['avg']:.4f}s P90={overhead['p90']:.4f}s"
            )
        if info["xfer_fail_total"]:
            lines.append(f"KV_XFER_FAIL 全局计数: {info['xfer_fail_total']}（无 request_id，未归属单请求）")
        lines.append("=" * 96)
        logging.info("\n".join(lines))

    def csv_extra_metrics(self) -> Dict[str, float]:
        """aisbench_result.csv 追加列（extra_metrics 机制）。"""
        info = self.summary()
        extra: Dict[str, float] = {}
        column_map = {
            "p_waiting": "p_waiting_avg_s",
            "p_running": "p_running_avg_s",
            "kv_transfer": "kv_transfer_avg_s",
            "d_waiting": "d_waiting_avg_s",
            "d_running": "d_running_avg_s",
        }
        for key, col in column_map.items():
            stage = info["stages"].get(key)
            extra[col] = round(stage["avg"], 6) if stage else ""
        return extra

    def write_details_csv(self, output_dir: str) -> Optional[str]:
        """请求明细 CSV（kv_stage_details.csv）。"""
        if not self.records:
            return None
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "kv_stage_details.csv")
        exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if not exists:
                writer.writerow([
                    "seq", "request_id", "phase", "client_id", "client_span",
                    "p_waiting_s", "p_running_s", "kv_transfer_s",
                    "d_waiting_s", "d_running_s", "kv_wait_overhead_s",
                    "worker_count", "worker_failed", "kv_bytes_total",
                    "kv_elapsed_ms_max", "kv_elapsed_ms_avg", "missing_p",
                ])
            for r in self.records:
                writer.writerow([
                    r.get("seq", ""), r["request_id"], r.get("phase", ""),
                    r.get("client_id", ""), round(r["client_span"], 3) if r.get("client_span") is not None else "",
                    r["p_waiting"], r["p_running"], r["kv_transfer"],
                    r["d_waiting"], r["d_running"], r["kv_wait_overhead"],
                    r["worker_count"], r["worker_failed"], r["kv_bytes_total"],
                    r["kv_elapsed_ms_max"],
                    round(r["kv_elapsed_ms_avg"], 3) if r["kv_elapsed_ms_avg"] is not None else "",
                    r["missing_p"],
                ])
        logging.info("[kv_stage] 请求明细已保存: %s", path)
        return path

    # ---------- 图 11 数据 ----------

    def build_bar_traces(self) -> Optional[Dict[str, Any]]:
        """构造图 11 水平分段时间线数据（绝对时间轴版，与全图 X 轴联动）。

        Y 轴 = 请求编号（完成顺序）；X 轴 = 绝对时间（date 轴，与图 1-9 联动框选）。
        各阶段起止由日志时间戳锚定反推（P 行时间戳 = P 完成锚点，D 行时间戳 = D 完成锚点）：
            p_running: [t_p_end - p_running, t_p_end]
            p_waiting: [t_p_end - p_running - p_waiting, ...]
            d_running: [t_d_end - d_running, t_d_end]
            d_waiting: [t_d_end - d_running - d_waiting, ...]
            kv_transfer: [t_d_end - d_running - d_waiting - kv_transfer, ...]
        注：vllm 日志时间戳为秒级精度，时间线存在 ≤1s 的取整误差；P 完成到 KV 开始的
        proxy 转发间隙由两个锚点差自然呈现。

        返回 {"y": [...], "stages": {key: {"base": [datetime], "width": [...], "text": [...]}},
               "sampled": bool, "total": int, "no_timestamp": int, "summary": {...}}
        """
        if not self.records:
            return None
        records = self.records
        sampled = False
        if len(records) > MAX_PLOT_REQUESTS:
            step = len(records) / MAX_PLOT_REQUESTS
            records = [records[int(i * step)] for i in range(MAX_PLOT_REQUESTS)]
            sampled = True

        def rid_short(rid: str) -> str:
            return ("…" + rid[-8:]) if len(rid) > 8 else rid

        # 照搬图 1 画图逻辑（代码证据 merge_aisbench_plots + 渲染证据 "Warmup TTFT" 虚线 Y=1-16）：
        # full 与 warmup 各自独立编号、都从 1 起、共用同一根 Request Index 轴（编号段重叠）。
        # 图 1 用线型区分（warmup 虚线 / full 实线），bar 场景由 monitor 用半透明区分。
        full_fallback_seen = 0
        warmup_seen = 0

        def y_label(rec: Dict[str, Any]) -> Any:
            nonlocal full_fallback_seen, warmup_seen
            if not self._is_full_phase(rec.get("phase")):
                warmup_seen += 1
                return warmup_seen
            if rec.get("client_id") is not None:
                return rec["client_id"] + 1
            full_fallback_seen += 1
            return full_fallback_seen

        def sub(ts: Optional[datetime], seconds: float) -> Optional[datetime]:
            return ts - timedelta(seconds=seconds) if ts is not None else None

        y_labels: List[Any] = []
        warmup_y_labels: List[Any] = []
        stage_base: Dict[str, List[Optional[datetime]]] = {key: [] for key in STAGE_KEYS}
        stage_width: Dict[str, List[float]] = {key: [] for key in STAGE_KEYS}
        hover: Dict[str, List[str]] = {key: [] for key in STAGE_KEYS}
        warmup_stage_base: Dict[str, List[Optional[datetime]]] = {key: [] for key in STAGE_KEYS}
        warmup_stage_width: Dict[str, List[float]] = {key: [] for key in STAGE_KEYS}
        warmup_hover: Dict[str, List[str]] = {key: [] for key in STAGE_KEYS}
        no_ts_count = 0
        # 批内任一有效时间锚（供时间戳缺失的记录放置不可见零宽 bar）
        fallback_ts: Optional[datetime] = None
        for rec in records:
            for key in ("t_p_end", "t_d_end"):
                if rec.get(key) and (fallback_ts is None or rec[key] < fallback_ts):
                    fallback_ts = rec[key]

        for idx, rec in enumerate(records, start=1):
            seq = rec.get("seq") or idx
            label = y_label(rec)
            is_warmup = not self._is_full_phase(rec.get("phase"))
            if is_warmup:
                warmup_y_labels.append(label)
            else:
                y_labels.append(label)
            values = {key: (rec.get(key) if rec.get(key) is not None else 0.0) for key in STAGE_KEYS}
            t_p_end, t_d_end = rec.get("t_p_end"), rec.get("t_d_end")
            has_ts = t_p_end is not None or t_d_end is not None
            if not has_ts:
                no_ts_count += 1

            # 各阶段 [base, width]（时间戳缺失时全段宽 0，不参与时间轴显示）
            d_run = t_d_end and has_ts
            p_run = t_p_end and has_ts
            anchor = fallback_ts or datetime.now()
            kv_base = sub(t_d_end, values["d_running"] + values["d_waiting"] + values["kv_transfer"]) if d_run else anchor
            p_end = t_p_end if p_run else None
            # 钳制：KV 拉取不可能早于 P 执行结束。秒级日志锚点取整可能造成反推起点
            # 早于 P 完成（视觉上 KV 段覆盖 P 执行段），按业务顺序钳到 P 完成时刻。
            gap_p_to_kv = None
            if kv_base is not None and p_end is not None:
                gap_p_to_kv = (kv_base - p_end).total_seconds()
                if kv_base < p_end:
                    kv_base = p_end
            segments = {
                "p_waiting": (sub(t_p_end, values["p_running"] + values["p_waiting"]) if p_run else anchor,
                              values["p_waiting"] if p_run else 0.0),
                "p_running": (sub(t_p_end, values["p_running"]) if p_run else anchor,
                              values["p_running"] if p_run else 0.0),
                "kv_transfer": (kv_base, values["kv_transfer"] if d_run else 0.0),
                "d_waiting": (sub(t_d_end, values["d_running"] + values["d_waiting"]) if d_run else anchor,
                              values["d_waiting"] if d_run else 0.0),
                "d_running": (sub(t_d_end, values["d_running"]) if d_run else anchor,
                              values["d_running"] if d_run else 0.0),
            }
            total = sum(values.values())
            phase_note = " [warmup]" if not self._is_full_phase(rec.get("phase")) else ""
            missing_note = " [P 日志缺失]" if rec.get("missing_p") else ""
            no_ts_note = " [日志时间戳缺失，未上图]" if not has_ts else ""
            head = f"#{label}（req {rid_short(rec['request_id'])}）总时延 {total:.3f}s{phase_note}{missing_note}{no_ts_note}"
            stage_line = " → ".join(
                f"{STAGE_LABELS[key]} {values[key]:.3f}s" for key in STAGE_KEYS
            )
            time_line = ""
            if has_ts:
                parts_t = []
                if t_p_end is not None:
                    parts_t.append(f"P 完成 {t_p_end.strftime('%H:%M:%S')}")
                if t_d_end is not None:
                    parts_t.append(f"D 完成 {t_d_end.strftime('%H:%M:%S')}")
                if gap_p_to_kv is not None:
                    parts_t.append(f"P→KV 间隙 {gap_p_to_kv:.3f}s")
                time_line = "<br>时间锚点: " + "，".join(parts_t)
            kv_line = ""
            if rec.get("kv_elapsed_ms_max") is not None:
                mb = rec["kv_bytes_total"] / (1024 * 1024)
                failed = rec["worker_failed"]
                fail_note = f" / ≈{failed} 失败" if failed else ""
                kv_line = (
                    f"<br>KV: 纯传输 {rec['kv_elapsed_ms_max']:.1f}ms | "
                    f"等待开销 ≈{rec['kv_wait_overhead']:.3f}s | "
                    f"worker {rec['worker_count']} 成功{fail_note} | 数据量 {mb:.1f}MB"
                )
            text = f"{head}<br>{stage_line}{time_line}{kv_line}"
            tgt_base = warmup_stage_base if is_warmup else stage_base
            tgt_width = warmup_stage_width if is_warmup else stage_width
            tgt_hover = warmup_hover if is_warmup else hover
            for key in STAGE_KEYS:
                base, width = segments[key]
                tgt_base[key].append(base if base is not None else anchor)
                tgt_width[key].append(width)
                tgt_hover[key].append(text)
        if no_ts_count:
            logging.warning("[kv_stage] %d 条请求日志时间戳缺失，未上图（CSV 数据完整）", no_ts_count)
        return {
            "y": y_labels,
            "stages": {key: {"base": stage_base[key], "width": stage_width[key], "text": hover[key]}
                       for key in STAGE_KEYS},
            "warmup_y": warmup_y_labels,
            "warmup_stages": {key: {"base": warmup_stage_base[key], "width": warmup_stage_width[key],
                                    "text": warmup_hover[key]} for key in STAGE_KEYS},
            "sampled": sampled,
            "total": len(self.records),
            "no_timestamp": no_ts_count,
            "summary": self.summary(),
        }
