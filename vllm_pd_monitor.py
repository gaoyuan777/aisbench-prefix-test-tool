# -*- coding: utf-8 -*-
"""vLLM 运行时指标采集：支持 PD 分离（P/D）与混部（M_LISTEN_SERVER），实时文件监控与 HTML 汇总。"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
import threading
import time
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover
    go = None
    make_subplots = None

try:
    import tabulate
except ImportError:  # pragma: no cover
    tabulate = None


POLL_INTERVAL_SEC = 1.0
REQUEST_TIMEOUT_SEC = 2.5
LIVE_FILE_NAME = "pd_metrics_live.txt"
EXP_FOLDER_RE = re.compile(r"Current exp folder:\s*(.+)$")
PROGRESS_BAR_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+)\] .*Starting progress bar"
)
SKIP_RESULT_KEYS = {"Prefix Cache Hit Rate", "Prefix Hit Rate", "prefix cache hit rate"}
CARTESIAN_CELLS = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 2))
RELAYOUT_SYNC_JS = """
(function() {
  var gd = document.getElementById('{plot_id}');
  if (!gd) return;
  var syncing = false;
  function cartesianXAxes() {
    return Object.keys(gd.layout || {}).filter(function(k) {
      if (!/^xaxis[0-9]*$/.test(k)) return false;
      var ax = gd.layout[k];
      return ax && ax.visible !== false && Array.isArray(ax.domain);
    });
  }
  gd.on('plotly_relayout', function(ev) {
    if (syncing || !ev) return;
    var axes = cartesianXAxes();
    var changed = null;
    for (var i = 0; i < axes.length; i++) {
      var ax = axes[i];
      if (ev[ax + '.range[0]'] !== undefined || ev[ax + '.range[1]'] !== undefined
          || ev[ax + '.range'] !== undefined || ev[ax + '.autorange'] !== undefined) {
        changed = ax;
        break;
      }
    }
    if (!changed) return;
    var src = gd.layout[changed];
    if (!src) return;
    var update = {};
    for (var j = 0; j < axes.length; j++) {
      var tgt = axes[j];
      if (tgt === changed) continue;
      if (src.autorange) {
        update[tgt + '.autorange'] = true;
      } else if (src.range) {
        update[tgt + '.range'] = src.range.slice();
        update[tgt + '.autorange'] = false;
      }
    }
    if (!Object.keys(update).length) return;
    syncing = true;
    Plotly.relayout(gd, update).then(function() { syncing = false; }, function() { syncing = false; });
  });
})();
"""

USAGE_METRICS = (
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "vllm:npu_cache_usage_perc",
)
HITS_METRICS = (
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_hits",
    "vllm:gpu_prefix_cache_hits_total",
    "vllm:gpu_prefix_cache_hits",
)
QUERIES_METRICS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_queries",
    "vllm:gpu_prefix_cache_queries_total",
    "vllm:gpu_prefix_cache_queries",
)
RUNNING_METRICS = ("vllm:num_requests_running",)
WAITING_METRICS = ("vllm:num_requests_waiting",)
SWAPPED_METRICS = ("vllm:num_requests_swapped",)
PREEMPT_METRICS = (
    "vllm:num_preemptions",
    "vllm:num_preemptions_total",
)
CACHE_INFO_METRICS = ("vllm:cache_config_info",)
BLOCK_COUNT_LABELS = (
    "num_gpu_blocks",
    "num_npu_blocks",
    "num_blocks",
    "num_cpu_blocks",
)
DIRECT_BLOCK_METRICS = (
    "vllm:num_gpu_blocks",
    "vllm:num_npu_blocks",
    "vllm:num_free_gpu_blocks",
    "vllm:num_free_npu_blocks",
)

_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"')
_LIVE_LOCK = threading.Lock()

AXIS_CONFIG = dict(
    showline=True,
    showgrid=True,
    showticklabels=True,
    gridwidth=0.5,
    gridcolor="rgba(211,211,211,0.5)",
    linecolor="black",
)

WEBGL_CONFIG = {
    "scrollZoom": True,
    "plotGlPixelRatio": 1,
    "showLink": False,
    "displaylogo": False,
    "responsive": True,
}

ROLE_COLORS = {
    "P": "#1f77b4",
    "D": "#ff7f0e",
    "M": "#2ca02c",
}
ROLE_LABELS = {
    "P": "P",
    "D": "D",
    "M": "混部",
}
DP_LINE_PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
)

EMPTY_METRICS = {
    "kv_cache_usage_pct": None,
    "free_blocks": None,
    "prefix_cache_hits": None,
    "prefix_cache_queries": None,
    "num_running": None,
    "num_waiting": None,
    "num_swapped": None,
    "num_preemptions_total": None,
    "num_gpu_blocks": None,
    "engines": {},
}


def _as_endpoint_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        text = str(value).strip()
        if not text:
            return []
        items = [part.strip() for part in text.split(",") if part.strip()]
    result = []
    for item in items:
        endpoint = str(item).strip().strip('"').strip("'")
        if endpoint:
            result.append(endpoint)
    return result


def role_label(role: Any) -> str:
    key = str(role or "")
    return ROLE_LABELS.get(key, key or "-")


def hit_rate_stage_label(item: Dict[str, Any], summary: Optional[List[Dict[str, Any]]] = None) -> str:
    role = item.get("role")
    if role == "M":
        peers = [row for row in (summary or []) if row.get("role") == "M"]
        if len(peers) > 1:
            return f"混部 {item.get('endpoint') or '-'}"
        return "混部"
    return role_label(role)


def parse_colocated_endpoints() -> List[str]:
    try:
        import config as cfg
    except ImportError:
        return []
    return _as_endpoint_list(getattr(cfg, "M_LISTEN_SERVER", "") or [])


def parse_role_endpoints() -> List[Tuple[str, str]]:
    """从 config.py 读取 P/D 监听端点。未配置的角色不会出现在结果中。"""
    try:
        import config as cfg
    except ImportError:
        return []

    roles = []
    for role, attr in (("P", "P_LISTEN_SERVER"), ("D", "D_LISTEN_SERVER")):
        for endpoint in _as_endpoint_list(getattr(cfg, attr, "")):
            roles.append((role, endpoint))
    return roles


def parse_monitor_config() -> Dict[str, Any]:
    """混部 M_LISTEN_SERVER 优先；未填时才走 P/D 分离监听。与 POD_INFO 互不改写。"""
    colocated = parse_colocated_endpoints()
    pd_endpoints = parse_role_endpoints()
    if colocated:
        return {
            "mode": "colocated",
            "endpoints": [("M", endpoint) for endpoint in colocated],
            "ignored_pd": pd_endpoints,
        }
    if pd_endpoints:
        return {"mode": "pd", "endpoints": pd_endpoints, "ignored_pd": []}
    return {"mode": "", "endpoints": [], "ignored_pd": []}


def split_host_port(endpoint: str) -> Tuple[str, str]:
    if endpoint.startswith("["):
        ip, port = endpoint.split("]:", 1)
        return ip[1:], port
    if endpoint.count(":") > 1:
        ip, port = endpoint.rsplit(":", 1)
        return ip, port
    ip, port = endpoint.split(":", 1)
    return ip, port


def _parse_labels(label_blob: str) -> Dict[str, str]:
    return {key: value.replace('\\"', '"') for key, value in _LABEL_RE.findall(label_blob)}


def parse_prometheus_text(text: str) -> List[Dict[str, Any]]:
    samples = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("}"):
            continue
        if "{" in line:
            name, rest = line.split("{", 1)
            if "}" not in rest:
                continue
            label_blob, value_blob = rest.split("}", 1)
            labels = _parse_labels(label_blob)
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            name, value_blob = parts[0], " ".join(parts[1:])
            labels = {}
        fields = value_blob.strip().split()
        if not fields:
            continue
        try:
            value = float(fields[0])
        except ValueError:
            continue
        samples.append({"name": name.strip(), "labels": labels, "value": value})
    return samples


def _engine_id(labels: Dict[str, str]) -> str:
    return labels.get("engine") or labels.get("engine_id") or "0"


def _pick_metric(samples: List[Dict[str, Any]], names: Iterable[str]) -> List[Dict[str, Any]]:
    wanted = set(names)
    return [item for item in samples if item["name"] in wanted]


def _to_usage_percent(raw_values: List[float]) -> Optional[float]:
    if not raw_values:
        return None
    avg = sum(raw_values) / len(raw_values)
    if avg <= 1.5:
        return avg * 100.0
    return avg


def extract_runtime_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把 /metrics 解析结果汇总成单端点快照。多 engine 时做聚合。"""
    usage_items = _pick_metric(samples, USAGE_METRICS)
    hits_items = _pick_metric(samples, HITS_METRICS)
    queries_items = _pick_metric(samples, QUERIES_METRICS)
    running_items = _pick_metric(samples, RUNNING_METRICS)
    waiting_items = _pick_metric(samples, WAITING_METRICS)
    swapped_items = _pick_metric(samples, SWAPPED_METRICS)
    preempt_items = []
    for name in PREEMPT_METRICS:
        preempt_items = _pick_metric(samples, (name,))
        if preempt_items:
            break
    if not preempt_items:
        preempt_items = [
            item for item in samples
            if "preempt" in item["name"].lower()
            and "bucket" not in item["name"].lower()
            and not item["name"].endswith("_sum")
            and not item["name"].endswith("_count")
        ]
    info_items = _pick_metric(samples, CACHE_INFO_METRICS)
    direct_block_items = _pick_metric(samples, DIRECT_BLOCK_METRICS)

    engine_blocks: Dict[str, float] = {}
    for item in info_items:
        for label in BLOCK_COUNT_LABELS:
            if label in item["labels"]:
                try:
                    engine_blocks[_engine_id(item["labels"])] = float(item["labels"][label])
                except ValueError:
                    continue
                break

    free_from_direct: List[float] = []
    total_from_direct: Dict[str, float] = {}
    for item in direct_block_items:
        engine = _engine_id(item["labels"])
        if "free" in item["name"]:
            free_from_direct.append(item["value"])
        else:
            total_from_direct[engine] = item["value"]
            engine_blocks.setdefault(engine, item["value"])

    usage_by_engine = {_engine_id(item["labels"]): item["value"] for item in usage_items}
    hits_by_engine = {_engine_id(item["labels"]): item["value"] for item in hits_items}
    queries_by_engine = {_engine_id(item["labels"]): item["value"] for item in queries_items}
    running_by_engine = {_engine_id(item["labels"]): item["value"] for item in running_items}
    waiting_by_engine = {_engine_id(item["labels"]): item["value"] for item in waiting_items}
    swapped_by_engine = {_engine_id(item["labels"]): item["value"] for item in swapped_items}
    preempt_by_engine = {_engine_id(item["labels"]): item["value"] for item in preempt_items}

    usage_pct = _to_usage_percent(list(usage_by_engine.values()))

    free_blocks = None
    total_blocks = sum(engine_blocks.values()) if engine_blocks else None
    if free_from_direct:
        free_blocks = sum(free_from_direct)
    elif engine_blocks and usage_by_engine:
        free = 0.0
        for engine, total in engine_blocks.items():
            raw = usage_by_engine.get(engine)
            if raw is None:
                continue
            frac = raw if raw <= 1.5 else raw / 100.0
            free += total * max(0.0, 1.0 - frac)
        free_blocks = free
    elif total_from_direct and usage_pct is not None:
        free_blocks = sum(total_from_direct.values()) * max(0.0, 1.0 - usage_pct / 100.0)

    return {
        "kv_cache_usage_pct": usage_pct,
        "free_blocks": free_blocks,
        "prefix_cache_hits": sum(hits_by_engine.values()) if hits_by_engine else None,
        "prefix_cache_queries": sum(queries_by_engine.values()) if queries_by_engine else None,
        "num_running": sum(running_by_engine.values()) if running_by_engine else None,
        "num_waiting": sum(waiting_by_engine.values()) if waiting_by_engine else None,
        "num_swapped": sum(swapped_by_engine.values()) if swapped_by_engine else None,
        "num_preemptions_total": sum(preempt_by_engine.values()) if preempt_by_engine else None,
        "num_gpu_blocks": total_blocks,
        "engines": {
            engine: {
                "kv_cache_usage_raw": usage_by_engine.get(engine),
                "prefix_cache_hits": hits_by_engine.get(engine),
                "prefix_cache_queries": queries_by_engine.get(engine),
                "num_running": running_by_engine.get(engine),
                "num_waiting": waiting_by_engine.get(engine),
                "num_gpu_blocks": engine_blocks.get(engine),
            }
            for engine in sorted(
                set(usage_by_engine)
                | set(hits_by_engine)
                | set(queries_by_engine)
                | set(running_by_engine)
                | set(waiting_by_engine)
                | set(engine_blocks)
            )
        },
    }


def fetch_metrics_text(endpoint: str) -> str:
    ip, port = split_host_port(endpoint)
    url = f"http://{ip}:{port}/metrics"
    resp = requests.get(url, proxies={"http": None, "https": None}, timeout=REQUEST_TIMEOUT_SEC)
    resp.raise_for_status()
    return resp.text


def collect_endpoint(role: str, endpoint: str) -> Dict[str, Any]:
    collected_at = datetime.now()
    try:
        text = fetch_metrics_text(endpoint)
        parsed = extract_runtime_metrics(parse_prometheus_text(text))
        error = None
    except Exception as exc:  # noqa: BLE001
        parsed = dict(EMPTY_METRICS)
        error = f"{type(exc).__name__}: {exc}"
    parsed.update(
        {
            "role": role,
            "endpoint": endpoint,
            "collected_at": collected_at.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "error": error,
        }
    )
    return parsed


def is_full_dataset_phase(phase: str) -> bool:
    return str(phase or "").startswith("full_dataset")


def compute_prefix_deltas(hits, queries, base_hits, base_queries) -> Dict[str, Optional[float]]:
    if None in (hits, queries, base_hits, base_queries):
        return {"prefix_cache_hits_delta": None, "prefix_cache_queries_delta": None, "prefix_hit_rate": None}
    hits_delta = hits - base_hits
    queries_delta = queries - base_queries
    hit_rate = (100.0 * hits_delta / queries_delta) if queries_delta > 0 else None
    return {
        "prefix_cache_hits_delta": hits_delta,
        "prefix_cache_queries_delta": queries_delta,
        "prefix_hit_rate": hit_rate,
    }


def _fmt_int(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{int(round(float(value)))}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_float(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def _tabulate(rows: List[List[Any]], headers: List[str]) -> str:
    if tabulate is not None:
        return tabulate.tabulate(
            rows,
            headers=headers,
            tablefmt="fancy_grid",
            numalign="center",
            stralign="left",
            missingval="N/A",
        )
    widths = [len(h) for h in headers]
    str_rows = []
    for row in rows:
        cells = [("N/A" if c is None else str(c)) for c in row]
        str_rows.append(cells)
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(cell))
    sep = "+".join("-" * (w + 2) for w in widths)
    sep = f"+{sep}+"
    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    lines = [sep, fmt_row(headers), sep]
    lines.extend(fmt_row(row) for row in str_rows)
    lines.append(sep)
    return "\n".join(lines)


def render_live_text(
    round_ts: datetime,
    phase: str,
    snapshots: List[Dict[str, Any]],
    live_path: str,
    baseline_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    mode: str = "pd",
) -> str:
    runtime_headers = [
        "Role",
        "Endpoint",
        "KVCache Usage (%)",
        "Free Blocks",
        "Running",
        "Waiting",
        "Swapped",
        "Preempts",
        "Status",
    ]
    runtime_rows = []
    hit_headers = [
        "Role",
        "Endpoint",
        "Hits Delta",
        "Queries Delta",
        "Hit Rate",
        "Hits (abs)",
        "Queries (abs)",
    ]
    hit_rows = []
    for item in snapshots:
        runtime_rows.append(
            [
                role_label(item.get("role")),
                item.get("endpoint"),
                _fmt_float(item.get("kv_cache_usage_pct"), 2),
                _fmt_int(item.get("free_blocks")),
                _fmt_int(item.get("num_running")),
                _fmt_int(item.get("num_waiting")),
                _fmt_int(item.get("num_swapped")),
                _fmt_int(item.get("num_preemptions_total")),
                item.get("error") or "ok",
            ]
        )
        hit_rows.append(
            [
                role_label(item.get("role")),
                item.get("endpoint"),
                _fmt_int(item.get("prefix_cache_hits_delta")),
                _fmt_int(item.get("prefix_cache_queries_delta")),
                _fmt_float(item.get("prefix_hit_rate"), 2, "%"),
                _fmt_int(item.get("prefix_cache_hits")),
                _fmt_int(item.get("prefix_cache_queries")),
            ]
        )

    mode_title = "vLLM 混部 Runtime Metrics" if mode == "colocated" else "vLLM PD Runtime Metrics"
    lines = [
        mode_title,
        f"Mode       : {'colocated (M_LISTEN_SERVER)' if mode == 'colocated' else 'pd (P/D_LISTEN_SERVER)'}",
        f"Updated at : {round_ts.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Phase      : {phase or '-'}",
        f"Live file  : {live_path}",
        "Monitor    : tail -F pd_metrics_live.txt",
        "",
        "Runtime gauges (refreshed in-place every 1s, this file is overwritten not appended)",
        _tabulate(runtime_rows, runtime_headers),
        "",
        "Prefix cache hit rate (full_dataset only; warmup is excluded)",
        "hit_rate = (hits_now - hits_baseline) / (queries_now - queries_baseline)",
        f"Baseline captured at : {baseline_ts or 'N/A (not in full_dataset yet)'}",
        f"End snapshot at      : {end_ts or '-'}",
        _tabulate(hit_rows, hit_headers),
        "",
    ]
    if not is_full_dataset_phase(phase):
        lines.append("Note: current phase is not full_dataset, Hit Rate columns stay N/A until full dataset starts.")
    return "\n".join(lines) + "\n"


def write_live_file(path: str, content: str) -> None:
    """覆盖写同一文件，便于 GNU tail -f / tail -F 检测截断后刷新。"""
    data = content.encode("utf-8")
    with _LIVE_LOCK:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(path):
            with open(path, "r+b") as fh:
                fh.seek(0)
                fh.write(data)
                fh.truncate()
        else:
            with open(path, "wb") as fh:
                fh.write(data)


def parse_last_exp_folder(log_path: str = "aisbench.log") -> Optional[str]:
    if not os.path.exists(log_path):
        return None
    last = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            match = EXP_FOLDER_RE.search(line.rstrip())
            if match:
                last = match.group(1).strip()
    return last


def resolve_plot_output_dir(output_dir: str, model_abbr: str, log_path: str = "aisbench.log") -> str:
    """定位 ais_bench plot 同级目录: <exp>/performances/<model_abbr>。"""
    exp = parse_last_exp_folder(log_path)
    if not exp:
        if os.path.isdir(output_dir):
            timed = [
                name for name in os.listdir(output_dir)
                if re.fullmatch(r"\d{8}_\d{6}", name) and os.path.isdir(os.path.join(output_dir, name))
            ]
            if timed:
                exp = os.path.join(output_dir, sorted(timed)[-1])
    if not exp:
        exp = output_dir
    perf_dir = os.path.join(exp, "performances", model_abbr)
    os.makedirs(perf_dir, exist_ok=True)
    return perf_dir


class PDMetricsMonitor:
    def __init__(
        self,
        endpoints: List[Tuple[str, str]],
        interval: float = POLL_INTERVAL_SEC,
        live_file: Optional[str] = None,
        mode: str = "pd",
        ignored_pd: Optional[List[Tuple[str, str]]] = None,
    ):
        self.endpoints = endpoints
        self.interval = interval
        self.mode = mode or ("pd" if endpoints else "")
        self.ignored_pd = ignored_pd or []
        self.enabled = bool(endpoints)
        self.phase = ""
        self.live_file_path = os.path.abspath(live_file or os.path.join(os.getcwd(), LIVE_FILE_NAME))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: List[Dict[str, Any]] = []
        self._phases: List[Dict[str, str]] = []
        self._lock = threading.Lock()
        self._warn_once = set()
        self._full_baseline: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._baseline_ts: Optional[str] = None
        self._end_ts: Optional[str] = None
        self._latest_snapshots: List[Dict[str, Any]] = []
        self.started_at: Optional[datetime] = None
        self.stopped_at: Optional[datetime] = None
        self._aisbench_plots: List[Dict[str, Any]] = []

    @classmethod
    def from_config(cls) -> "PDMetricsMonitor":
        cfg = parse_monitor_config()
        return cls(cfg["endpoints"], mode=cfg["mode"], ignored_pd=cfg.get("ignored_pd") or [])

    def set_phase(self, phase: str) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with self._lock:
            if self._phases and self._phases[-1]["end"] is None:
                self._phases[-1]["end"] = now
            self.phase = phase
            if phase:
                self._phases.append({"name": phase, "start": now, "end": None})
            if is_full_dataset_phase(phase):
                self._full_baseline = {}
                self._baseline_ts = None
                self._end_ts = None

    def remember_aisbench_plot(self, path: Optional[str], label: str = "") -> None:
        """ais_bench 每一阶段结束后立刻记下 gsm8k_plot，避免全量跑完后只剩最后一次。"""
        if not path or not os.path.exists(path):
            return
        origin = parse_last_progress_start("aisbench.log")
        if origin is None:
            origin = parse_last_progress_start(_exp_aisbench_log(path) or "")
        rec = {
            "path": os.path.abspath(path),
            "label": label or "",
            "origin": origin,
        }
        with self._lock:
            self._aisbench_plots = [item for item in self._aisbench_plots if item.get("path") != rec["path"]]
            self._aisbench_plots.append(rec)

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self.started_at = datetime.now()
        self._stop.clear()
        write_live_file(
            self.live_file_path,
            render_live_text(self.started_at, self.phase, [], self.live_file_path, mode=self.mode),
        )
        self._thread = threading.Thread(target=self._run, name="vllm-pd-metrics", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + REQUEST_TIMEOUT_SEC + 1)
            self._thread = None
        self.stopped_at = datetime.now()
        now = self.stopped_at.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with self._lock:
            if self._phases and self._phases[-1]["end"] is None:
                self._phases[-1]["end"] = now
            snapshots = list(self._latest_snapshots)
        if snapshots:
            write_live_file(
                self.live_file_path,
                render_live_text(
                    self.stopped_at,
                    self.phase,
                    snapshots,
                    self.live_file_path,
                    self._baseline_ts,
                    self._end_ts,
                    mode=self.mode,
                ),
            )

    def capture_full_dataset_baseline(self) -> List[Dict[str, Any]]:
        """全量数据集开始前打一次 metrics，作为命中率基准。"""
        round_ts = datetime.now()
        snapshots = self._fetch_all()
        aligned = round_ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with self._lock:
            self._full_baseline = {}
            self._baseline_ts = aligned
            self._end_ts = None
            for item in snapshots:
                self._full_baseline[(item["role"], item["endpoint"])] = {
                    "prefix_cache_hits": item.get("prefix_cache_hits"),
                    "prefix_cache_queries": item.get("prefix_cache_queries"),
                    "timestamp": aligned,
                }
            self._store_locked(snapshots, round_ts)
        write_live_file(
            self.live_file_path,
            render_live_text(round_ts, self.phase, snapshots, self.live_file_path, self._baseline_ts, self._end_ts, mode=self.mode),
        )
        return snapshots

    def capture_full_dataset_end(self) -> List[Dict[str, Any]]:
        """全量数据集结束后再打一次 metrics，计算最终命中/查询增量与比值。"""
        round_ts = datetime.now()
        snapshots = self._fetch_all()
        with self._lock:
            self._end_ts = round_ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self._store_locked(snapshots, round_ts)
        write_live_file(
            self.live_file_path,
            render_live_text(round_ts, self.phase, snapshots, self.live_file_path, self._baseline_ts, self._end_ts, mode=self.mode),
        )
        return snapshots

    def _run(self) -> None:
        while not self._stop.is_set():
            round_started = time.perf_counter()
            round_ts = datetime.now()
            snapshots = self._poll_round(round_ts)
            write_live_file(
                self.live_file_path,
                render_live_text(round_ts, self.phase, snapshots, self.live_file_path, self._baseline_ts, self._end_ts, mode=self.mode),
            )
            remain = self.interval - (time.perf_counter() - round_started)
            if remain > 0:
                self._stop.wait(remain)

    def _fetch_all(self) -> List[Dict[str, Any]]:
        snapshots: List[Dict[str, Any]] = []
        if not self.endpoints:
            return snapshots
        with ThreadPoolExecutor(max_workers=max(1, len(self.endpoints))) as pool:
            futures = [pool.submit(collect_endpoint, role, endpoint) for role, endpoint in self.endpoints]
            for future in as_completed(futures):
                snapshots.append(future.result())
        snapshots.sort(key=lambda item: (0 if item["role"] == "P" else 1, item["endpoint"]))
        return snapshots

    def _apply_baseline_locked(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if is_full_dataset_phase(self.phase):
            base = self._full_baseline.get((item["role"], item["endpoint"]))
            if base:
                item.update(
                    compute_prefix_deltas(
                        item.get("prefix_cache_hits"),
                        item.get("prefix_cache_queries"),
                        base.get("prefix_cache_hits"),
                        base.get("prefix_cache_queries"),
                    )
                )
                return item
        item["prefix_cache_hits_delta"] = None
        item["prefix_cache_queries_delta"] = None
        item["prefix_hit_rate"] = None
        return item

    def _store_locked(self, snapshots: List[Dict[str, Any]], round_ts: datetime) -> None:
        aligned = round_ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        epoch = round_ts.timestamp()
        stored = []
        for item in snapshots:
            if item.get("error"):
                key = f"{item['role']}:{item['endpoint']}"
                if key not in self._warn_once:
                    self._warn_once.add(key)
                    sys.stdout.write(f"[PD-METRICS] 首次采集失败 {key}: {item['error']}\n")
                    sys.stdout.flush()
            record = dict(item)
            record["timestamp"] = aligned
            record["timestamp_epoch"] = epoch
            record["phase"] = self.phase
            self._apply_baseline_locked(record)
            self._samples.append(record)
            stored.append(record)
        self._latest_snapshots = stored
        snapshots[:] = stored

    def _poll_round(self, round_ts: datetime) -> List[Dict[str, Any]]:
        snapshots = self._fetch_all()
        with self._lock:
            self._store_locked(snapshots, round_ts)
        return snapshots

    def get_hit_rate_summary(self) -> List[Dict[str, Any]]:
        with self._lock:
            snapshots = list(self._latest_snapshots)
        summary = []
        for item in snapshots:
            summary.append(
                {
                    "role": item.get("role"),
                    "endpoint": item.get("endpoint"),
                    "hits_delta": item.get("prefix_cache_hits_delta"),
                    "queries_delta": item.get("prefix_cache_queries_delta"),
                    "hit_rate": item.get("prefix_hit_rate"),
                }
            )
        return summary

    def format_hit_rate_for_csv(self) -> str:
        summary = self.get_hit_rate_summary()
        parts = []
        for item in summary:
            rate = item.get("hit_rate")
            rate_text = f"{rate:.2f}%" if rate is not None else "N/A"
            parts.append(f"{hit_rate_stage_label(item, summary)}:{rate_text}")
        return "; ".join(parts) if parts else "N/A"

    def print_hit_rate_table(self) -> None:
        summary = self.get_hit_rate_summary()
        rows = [["Common Metric", "Stage", "Value"]]
        for item in summary:
            rate = item.get("hit_rate")
            rate_text = f"{rate:.2f}%" if rate is not None else "N/A"
            rows.append(["Prefix Cache Hit Rate", hit_rate_stage_label(item, summary), rate_text])
        if len(rows) == 1:
            rows.append(["Prefix Cache Hit Rate", "-", "N/A"])
        table = _tabulate(rows[1:], rows[0])
        logging.info("Performance Results extra row [Prefix Cache Hit Rate]:")
        sys.stdout.write("\nPerformance Results extra row [Prefix Cache Hit Rate]:\n" + table + "\n")
        sys.stdout.flush()

    def export(self, output_dir: str, run_tag: Optional[str] = None) -> Dict[str, str]:
        if not self.enabled:
            return {}
        os.makedirs(output_dir, exist_ok=True)
        jsonl_path = os.path.join(output_dir, "vllm_pd_runtime_metrics.jsonl")
        html_path = os.path.join(output_dir, "vllm_pd_runtime_metrics.html")
        plot_path = find_aisbench_plot_html(output_dir)
        with self._lock:
            samples = list(self._samples)
            phases = list(self._phases)
            baseline_ts = self._baseline_ts
            end_ts = self._end_ts
            aisbench_plots = list(self._aisbench_plots)
        aisbench_plots = merge_aisbench_plots(
            aisbench_plots,
            output_dir,
            self.started_at,
            self.stopped_at,
            fallback_path=plot_path,
        )
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for row in samples:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        hit_rate_summary = self.get_hit_rate_summary()
        write_metrics_html(
            html_path,
            samples,
            phases,
            self.started_at,
            self.stopped_at,
            baseline_ts=baseline_ts,
            end_ts=end_ts,
            aisbench_plot_path=plot_path,
            aisbench_plots=aisbench_plots,
            hit_rate_summary=hit_rate_summary,
            aisbench_log_path="aisbench.log",
            mode=self.mode,
        )
        append_hit_rate_to_aisbench_json(output_dir, hit_rate_summary)
        del run_tag
        return {"html": html_path, "jsonl": jsonl_path, "plot": plot_path or ""}


def _to_dt(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


BIN_SEC = 1.0
QUEUE_BAR_WIDTH_MS = 800
RUNNING_COLOR = "#1f77b4"
WAITING_COLOR = "#d62728"
PREEMPTED_COLOR = "#f0ad4e"


def find_aisbench_plot_html(perf_dir: str) -> Optional[str]:
    if not perf_dir or not os.path.isdir(perf_dir):
        return None
    preferred = os.path.join(perf_dir, "gsm8k_plot.html")
    if os.path.exists(preferred):
        return preferred
    found = []
    for name in os.listdir(perf_dir):
        low = name.lower()
        if not low.endswith("_plot.html"):
            continue
        if "rps_distribution" in low or "vllm_pd" in low:
            continue
        found.append(os.path.join(perf_dir, name))
    return sorted(found)[-1] if found else None


def plot_label_from_phase(phase: str) -> str:
    text = str(phase or "")
    if "warmup" in text.lower():
        return "Warmup"
    if is_full_dataset_phase(text):
        return "全量"
    return text


def discover_session_plots(
    perf_dir: str,
    started_at: Optional[datetime],
    stopped_at: Optional[datetime],
) -> List[str]:
    """在本次 monitor 时间窗内找出所有 gsm8k_plot.html（含 warmup 那次实验目录）。"""
    if not perf_dir or not started_at:
        return []
    start = started_at.timestamp() - 5
    end = (stopped_at or datetime.now()).timestamp() + 5
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(perf_dir))))
    if not os.path.isdir(root):
        return []
    found = []
    for ts_name in os.listdir(root):
        if not re.fullmatch(r"\d{8}_\d{6}", ts_name):
            continue
        perf_root = os.path.join(root, ts_name, "performances")
        if not os.path.isdir(perf_root):
            continue
        for model in os.listdir(perf_root):
            path = os.path.join(perf_root, model, "gsm8k_plot.html")
            if not os.path.isfile(path):
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if start <= mtime <= end:
                found.append(os.path.abspath(path))
    return sorted(set(found), key=os.path.getmtime)


def merge_aisbench_plots(
    remembered: List[Dict[str, Any]],
    perf_dir: str,
    started_at: Optional[datetime],
    stopped_at: Optional[datetime],
    fallback_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    plots = [dict(item) for item in remembered if item.get("path") and os.path.exists(item["path"])]
    known = {item["path"] for item in plots}
    for path in discover_session_plots(perf_dir, started_at, stopped_at):
        if path not in known:
            plots.append({"path": path, "label": "", "origin": None})
            known.add(path)
    if fallback_path and os.path.exists(fallback_path) and os.path.abspath(fallback_path) not in known:
        plots.append({"path": os.path.abspath(fallback_path), "label": "", "origin": None})
    plots.sort(key=lambda item: os.path.getmtime(item["path"]) if os.path.exists(item["path"]) else 0)
    if len(plots) >= 2:
        if not plots[0].get("label"):
            plots[0]["label"] = "Warmup"
        if not plots[-1].get("label"):
            plots[-1]["label"] = "全量"
        for idx, item in enumerate(plots[1:-1], start=2):
            if not item.get("label"):
                item["label"] = f"第{idx}次"
    elif len(plots) == 1 and not plots[0].get("label"):
        plots[0]["label"] = ""
    return plots


def _extract_balanced(text: str, start: int) -> Tuple[str, int]:
    opener = text[start]
    closer = {"[": "]", "{": "}"}[opener]
    depth = 0
    in_str = False
    escape = False
    quote = ""
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_str = False
            continue
        if ch in ('"', "'"):
            in_str = True
            quote = ch
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1], i + 1
    raise ValueError("unbalanced JSON in plot html")


def load_plotly_spec_from_html(path: str) -> Optional[Dict[str, Any]]:
    json_path = os.path.splitext(path)[0] + ".json"
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    with open(path, "r", encoding="utf-8") as fh:
        html = fh.read()
    marker = "Plotly.newPlot("
    idx = html.find(marker)
    if idx < 0:
        idx = html.find("Plotly.react(")
        if idx < 0:
            return None
        marker = "Plotly.react("
    cursor = idx + len(marker)
    while cursor < len(html) and html[cursor] in " \n\r\t":
        cursor += 1
    # skip the div id argument
    if html[cursor] in ('"', "'"):
        quote = html[cursor]
        cursor = html.find(quote, cursor + 1) + 1
        comma = html.find(",", cursor)
        cursor = comma + 1
    while cursor < len(html) and html[cursor] in " \n\r\t":
        cursor += 1
    data_raw, cursor = _extract_balanced(html, cursor)
    while cursor < len(html) and html[cursor] in " \n\r\t,":
        cursor += 1
    layout_raw, _ = _extract_balanced(html, cursor)
    data_raw = data_raw.replace("NaN", "null").replace("Infinity", "null")
    layout_raw = layout_raw.replace("NaN", "null").replace("Infinity", "null")
    return {"data": json.loads(data_raw), "layout": json.loads(layout_raw)}


def split_aisbench_plot_traces(spec: Dict[str, Any]) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
    first, second = [], []
    for trace in spec.get("data") or []:
        yaxis = str(trace.get("yaxis") or "y")
        cleaned = dict(trace)
        cleaned.pop("xaxis", None)
        cleaned.pop("yaxis", None)
        if yaxis in ("y", "y1"):
            first.append(cleaned)
        else:
            second.append(cleaned)
    if not second and first:
        # 非流式场景 gsm8k_plot 只有并发图，放到图3
        second, first = first, []
    return first, second, spec.get("layout") or {}


def _layout_axis_title(layout: Dict[str, Any], axis_key: str, default: str) -> str:
    axis = layout.get(axis_key) or {}
    title = axis.get("title")
    if isinstance(title, dict):
        return title.get("text") or default
    if isinstance(title, str) and title:
        return title
    return default


def bin_queue_series(
    samples: List[Dict[str, Any]], role: str
) -> Tuple[List[datetime], List[float], List[float], List[float], List[str], bool]:
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    role_samples = []
    for sample in samples:
        if sample.get("role") != role:
            continue
        role_samples.append(sample)
        epoch = sample.get("timestamp_epoch")
        if epoch is None:
            dt = _to_dt(sample.get("timestamp"))
            epoch = dt.timestamp() if dt else None
        if epoch is None:
            continue
        buckets.setdefault(int(epoch // BIN_SEC), []).append(sample)
    has_swapped = any(item.get("num_swapped") is not None for item in role_samples)
    has_preempt_counter = any(item.get("num_preemptions_total") is not None for item in role_samples)
    show_preempted = has_swapped or has_preempt_counter
    xs, running, waiting, preempted, hover = [], [], [], [], []
    prev_preempt: Dict[str, float] = {}
    for key in sorted(buckets):
        group = buckets[key]
        start_epoch = key * BIN_SEC
        start_dt = datetime.fromtimestamp(start_epoch)
        xs.append(start_dt)
        last_by_endpoint: Dict[str, Dict[str, Any]] = {}
        for sample in group:
            last_by_endpoint[str(sample.get("endpoint") or "")] = sample
        run_v = sum(float(item.get("num_running") or 0) for item in last_by_endpoint.values())
        wait_v = sum(float(item.get("num_waiting") or 0) for item in last_by_endpoint.values())
        preempt_v = 0.0
        used_events = False
        for endpoint, item in last_by_endpoint.items():
            swapped = item.get("num_swapped")
            total = item.get("num_preemptions_total")
            if swapped is not None and float(swapped) > 0:
                preempt_v += float(swapped)
            elif total is not None:
                prev = prev_preempt.get(endpoint)
                if prev is not None:
                    delta = max(0.0, float(total) - prev)
                    preempt_v += delta
                    if delta:
                        used_events = True
                prev_preempt[endpoint] = float(total)
            elif swapped is not None:
                preempt_v += float(swapped)
        running.append(run_v)
        waiting.append(wait_v)
        preempted.append(preempt_v)
        preempt_note = "本秒抢占次数" if used_events or (has_preempt_counter and not has_swapped) else "swapped占用"
        hover.append(
            f"角色: {role_label(role)}<br>"
            f"时间窗: {start_dt.strftime('%H:%M:%S')} ~ "
            f"{datetime.fromtimestamp(start_epoch + BIN_SEC).strftime('%H:%M:%S')}<br>"
            f"Running: {int(run_v)}<br>Waiting: {int(wait_v)}<br>"
            f"Running (preempted): {int(preempt_v)}（{preempt_note}）<br>"
            f"合计: {int(run_v + wait_v + preempt_v)}"
        )
    return xs, running, waiting, preempted, hover, show_preempted


def parse_last_progress_start(log_path: str) -> Optional[datetime]:
    if not log_path or not os.path.exists(log_path):
        return None
    last = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            match = PROGRESS_BAR_RE.search(line)
            if not match:
                continue
            try:
                last = datetime.strptime(f"{match.group(1)}.{match.group(2)}", "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                try:
                    last = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
    return last


def _exp_aisbench_log(perf_or_html_path: Optional[str]) -> Optional[str]:
    if not perf_or_html_path:
        return None
    path = os.path.abspath(perf_or_html_path)
    if path.lower().endswith(".html"):
        path = os.path.dirname(path)
    # perf_dir = <exp>/performances/<model>
    exp_dir = os.path.dirname(os.path.dirname(path))
    candidate = os.path.join(exp_dir, "aisbench.log")
    return candidate if os.path.exists(candidate) else None


def infer_plot_origin(
    samples: List[Dict[str, Any]],
    phases: List[Dict[str, str]],
    started_at: Optional[datetime],
    aisbench_log_path: Optional[str] = None,
    aisbench_plot_path: Optional[str] = None,
) -> datetime:
    """gsm8k_plot 的 x=0 对应第一条请求发出时刻，转成墙钟时间。"""
    candidates = []
    if aisbench_plot_path:
        candidates.append(_exp_aisbench_log(aisbench_plot_path))
    if aisbench_log_path:
        candidates.append(aisbench_log_path)
    candidates.append("aisbench.log")
    for path in candidates:
        origin = parse_last_progress_start(path or "")
        if origin is not None:
            return origin
    for sample in samples:
        if not is_full_dataset_phase(sample.get("phase") or ""):
            continue
        running = sample.get("num_running") or 0
        waiting = sample.get("num_waiting") or 0
        if running or waiting:
            dt = _to_dt(sample.get("timestamp"))
            if dt is not None:
                return dt
    for phase in phases:
        if is_full_dataset_phase(phase.get("name") or ""):
            dt = _to_dt(phase.get("start"))
            if dt is not None:
                return dt
    if samples:
        dt = _to_dt(samples[0].get("timestamp"))
        if dt is not None:
            return dt
    return started_at or datetime.now()


_PLOTLY_STRUCT_DTYPES = {
    "f8": ("d", 8),
    "f4": ("f", 4),
    "i8": ("q", 8),
    "i4": ("i", 4),
    "i2": ("h", 2),
    "i1": ("b", 1),
    "u8": ("Q", 8),
    "u4": ("I", 4),
    "u2": ("H", 2),
    "u1": ("B", 1),
}


def decode_plotly_typed_array(value: Any) -> Any:
    """Plotly HTML 常把 x/y 编成 {dtype, bdata}，必须先解成 Python 列表。"""
    if isinstance(value, dict) and "bdata" in value:
        raw = base64.b64decode(value.get("bdata") or "")
        dtype = str(value.get("dtype") or "f8").strip().lstrip("<>|=")
        try:
            import numpy as np  # type: ignore

            arr = np.frombuffer(raw, dtype=dtype)
            shape = value.get("shape")
            if shape:
                arr = arr.reshape(shape)
            return arr.tolist()
        except Exception:
            fmt_size = _PLOTLY_STRUCT_DTYPES.get(dtype)
            if not fmt_size:
                return value
            fmt, size = fmt_size
            count = len(raw) // size
            import struct

            unpacked = struct.unpack("<" + fmt * count, raw[: count * size])
            return list(unpacked)
    if isinstance(value, list) and value and isinstance(value[0], dict) and "bdata" in value[0]:
        return [decode_plotly_typed_array(item) for item in value]
    return value


def _is_missing_x(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (datetime, timedelta)):
        return False
    if isinstance(value, str):
        text = value.strip().lower()
        return text in ("", "null", "nan", "none")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number != number  # NaN


def _shift_trace_x_to_datetime(trace: Dict[str, Any], origin: datetime) -> None:
    xs = decode_plotly_typed_array(trace.get("x"))
    if xs is None:
        return
    if isinstance(xs, (int, float, str, datetime)):
        xs = [xs]
    if isinstance(xs, dict):
        logging.warning("gsm8k_plot trace x is still a typed-array dict after decode, skip shift")
        return
    shifted = []
    for value in xs:
        if _is_missing_x(value):
            shifted.append(None)
            continue
        if isinstance(value, datetime):
            shifted.append(value)
            continue
        if isinstance(value, str):
            parsed = _to_dt(value)
            if parsed is not None:
                shifted.append(parsed)
                continue
            if "T" in value or ":" in value:
                shifted.append(value)
                continue
            try:
                value = float(value)
            except ValueError:
                shifted.append(value)
                continue
        try:
            shifted.append(origin + timedelta(seconds=float(value)))
        except (TypeError, ValueError, OverflowError):
            shifted.append(value)
    trace["x"] = shifted


def _trace_line_color(trace: Dict[str, Any]) -> str:
    line = trace.get("line") or {}
    if isinstance(line, dict) and line.get("color"):
        return str(line.get("color")).lower()
    marker = trace.get("marker") or {}
    if isinstance(marker, dict) and marker.get("color"):
        return str(marker.get("color")).lower()
    return ""


def _copied_trace_name(trace: Dict[str, Any]) -> str:
    existing = str(trace.get("name") or "").strip()
    if existing and existing.lower() not in ("trace 0", "trace 1"):
        return existing
    color = _trace_line_color(trace)
    fill = str(trace.get("fill") or "").lower()
    if "red" in color:
        return "TTFT"
    if "blue" in color:
        return "Decode"
    if "4caf50" in color or "green" in color or fill == "tozeroy":
        return "Concurrency"
    return existing or "请求"


def _attach_copied_traces(
    fig,
    traces: List[dict],
    row: int,
    col: int,
    legend: str,
    origin: datetime,
    name_prefix: str = "",
    line_dash: Optional[str] = None,
) -> None:
    seen = set()
    for raw in traces:
        trace = dict(raw)
        trace.pop("xaxis", None)
        trace.pop("yaxis", None)
        if str(trace.get("type") or "").lower() == "scattergl":
            trace["type"] = "scatter"
        for key in ("y", "z", "text", "customdata", "r", "theta"):
            if key in trace:
                trace[key] = decode_plotly_typed_array(trace[key])
        _shift_trace_x_to_datetime(trace, origin)
        name = _copied_trace_name(trace)
        if name_prefix:
            name = f"{name_prefix}{name}"
        if line_dash:
            line = dict(trace.get("line") or {})
            line["dash"] = line_dash
            trace["line"] = line
        trace["name"] = name
        trace["legend"] = legend
        trace["legendgroup"] = f"{legend}-{name}"
        trace["showlegend"] = name not in seen
        seen.add(name)
        fig.add_trace(trace, row=row, col=col)


def format_hit_rate_parenthetical(summary: Optional[List[Dict[str, Any]]]) -> str:
    if not summary:
        return "(Prefix Cache Hit Rate: N/A)"
    parts = []
    for item in summary:
        rate = item.get("hit_rate")
        text = f"{rate:.2f}%" if isinstance(rate, (int, float)) else "N/A"
        parts.append(f"{hit_rate_stage_label(item, summary)} {text}")
    return "(Prefix Cache Hit Rate: " + "; ".join(parts) + ")" if parts else "(Prefix Cache Hit Rate: N/A)"


def _drop_skipped_result_columns(rows: Optional[List[List[str]]]) -> Optional[List[List[str]]]:
    if not rows:
        return rows
    header = [str(cell).strip() for cell in rows[0]]
    drop = {idx for idx, name in enumerate(header) if name in SKIP_RESULT_KEYS}
    if not drop:
        return rows
    keep = [idx for idx in range(len(header)) if idx not in drop]
    return [[(row[idx] if idx < len(row) else "") for idx in keep] for row in rows]


def load_aisbench_result_tables(perf_dir: Optional[str]) -> Tuple[Optional[List[List[str]]], Optional[List[List[str]]]]:
    if not perf_dir or not os.path.isdir(perf_dir):
        return None, None
    csv_rows = None
    json_rows = None
    for name in sorted(os.listdir(perf_dir)):
        low = name.lower()
        if "rps_distribution" in low or "vllm_pd" in low or "details" in low:
            continue
        path = os.path.join(perf_dir, name)
        if name.endswith(".csv") and csv_rows is None:
            try:
                with open(path, "r", encoding="utf-8-sig", newline="") as fh:
                    csv_rows = _drop_skipped_result_columns([list(row) for row in csv.reader(fh)])
            except Exception:
                csv_rows = None
        elif name.endswith(".json") and json_rows is None:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            rows = [["Common Metric", "Stage", "Value"]]
            for key, stage_value in data.items():
                if str(key).strip() in SKIP_RESULT_KEYS:
                    continue
                if not isinstance(stage_value, dict):
                    rows.append([str(key), "", str(stage_value)])
                    continue
                for stage_name, value in stage_value.items():
                    rows.append([str(key), str(stage_name), str(value)])
            json_rows = rows if len(rows) > 1 else None
    return csv_rows, json_rows


def _pad_row(row: List[Any], width: int) -> List[str]:
    cells = ["" if c is None else str(c) for c in row]
    if len(cells) < width:
        cells.extend([""] * (width - len(cells)))
    return cells[:width]


def _build_result_table_trace(csv_rows, json_rows) -> Any:
    sections: List[Tuple[str, List[List[str]]]] = []
    if csv_rows:
        sections.append(("Performance Parameters", csv_rows))
    if json_rows:
        sections.append(("Common Metric", json_rows))
    if not sections:
        return go.Table(
            header=dict(
                values=["提示"],
                fill_color="#1f4e79",
                font=dict(color="white", size=11),
                align="center",
            ),
            cells=dict(
                values=[["未找到 ais_bench 测试结果（gsm8k.csv / gsm8k.json）"]],
                fill_color="#f7f9fc",
                align="left",
                font=dict(size=11),
                height=24,
            ),
        )

    width = max(len(row) for _, rows in sections for row in rows)
    header_vals = _pad_row(sections[0][1][0], width) if sections[0][1] else _pad_row([sections[0][0]], width)
    body_rows: List[List[str]] = []
    row_kinds: List[str] = []
    for section_idx, (title, rows) in enumerate(sections):
        data_rows = rows[1:] if rows else []
        if section_idx > 0:
            body_rows.append(_pad_row([], width))
            row_kinds.append("blank")
            body_rows.append(_pad_row(rows[0] if rows else [title], width))
            row_kinds.append("header")
        for row in data_rows:
            body_rows.append(_pad_row(row, width))
            row_kinds.append("cell")
    if not body_rows:
        body_rows = [_pad_row(["-"], width)]
        row_kinds = ["cell"]

    cols = [[row[i] for row in body_rows] for i in range(width)]
    fills = []
    fonts = []
    cell_i = 0
    for kind in row_kinds:
        if kind == "header":
            fills.append("#1f4e79")
            fonts.append("white")
        elif kind == "blank":
            fills.append("#ffffff")
            fonts.append("#222222")
        else:
            fills.append("#ffffff" if cell_i % 2 else "#f4f7fb")
            fonts.append("#222222")
            cell_i += 1
    return go.Table(
        header=dict(
            values=header_vals,
            fill_color="#1f4e79",
            font=dict(color="white", size=11),
            align="center",
            height=28,
            line_color="#1f4e79",
        ),
        cells=dict(
            values=cols,
            fill_color=[fills for _ in range(width)],
            font=dict(color=[fonts for _ in range(width)], size=10),
            align=["left"] + ["center"] * (width - 1),
            height=24,
            line_color="#d0d7de",
        ),
    )


def _ordered_cartesian_axis_pairs(fig) -> List[Tuple[str, str]]:
    layout_json = fig.to_plotly_json().get("layout") or {}
    xkeys = [k for k in layout_json if re.fullmatch(r"xaxis\d*", str(k))]

    def axis_index(name: str) -> int:
        suffix = name.replace("xaxis", "")
        return 1 if suffix == "" else int(suffix)

    pairs = []
    for xk in sorted(xkeys, key=axis_index):
        axis = layout_json.get(xk) or {}
        if axis.get("visible") is False:
            continue
        if not axis.get("domain"):
            continue
        idx = axis_index(xk)
        yk = "yaxis" if idx == 1 else f"yaxis{idx}"
        pairs.append((xk, yk))
    return pairs


def _legend_style(x: float, y: float, xanchor: str, yanchor: str) -> dict:
    return dict(
        x=x,
        y=y,
        xanchor=xanchor,
        yanchor=yanchor,
        bgcolor="rgba(255,255,255,0.82)",
        bordercolor="#d0d7de",
        borderwidth=1,
        font=dict(size=10),
        itemsizing="constant",
        itemwidth=30,
        tracegroupgap=2,
        orientation="v",
    )


def _adaptive_legend_box(fig, xaxis_key: str, yaxis_key: str, ys: List[Any], prefer: str = "topright") -> dict:
    domain_x = list(fig.layout[xaxis_key].domain or [0, 1])
    domain_y = list(fig.layout[yaxis_key].domain or [0, 1])
    x0, x1 = domain_x[0], domain_x[1]
    y0, y1 = domain_y[0], domain_y[1]
    numeric = []
    for value in ys:
        try:
            if value is None:
                continue
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    corner = prefer
    if numeric:
        max_y = max(numeric) or 1.0
        n = len(numeric)
        right = numeric[int(n * 0.7) :] or numeric
        left = numeric[: max(1, int(n * 0.3))]
        right_busy = (sum(right) / len(right)) > 0.62 * max_y
        left_busy = (sum(left) / len(left)) > 0.62 * max_y
        if prefer == "topright" and right_busy and not left_busy:
            corner = "topleft"
        elif prefer == "topleft" and left_busy and not right_busy:
            corner = "topright"
        elif right_busy and left_busy:
            corner = "bottomright"
    frac = {
        "topright": (0.98, 0.97, "right", "top"),
        "topleft": (0.02, 0.97, "left", "top"),
        "bottomright": (0.98, 0.06, "right", "bottom"),
        "bottomleft": (0.02, 0.06, "left", "bottom"),
    }[corner]
    xf, yf, xanchor, yanchor = frac
    return _legend_style(x0 + (x1 - x0) * xf, y0 + (y1 - y0) * yf, xanchor, yanchor)


def _add_kv_lines(fig, samples: List[Dict[str, Any]], row: int, col: int, legend: str) -> None:
    """图2: 按角色聚合的 KV Cache 平均占用率折线。PD 分离时 P/D 各一条；per-DP 明细见图5/图8。"""
    role_order: List[str] = []
    role_endpoints: Dict[str, List[str]] = {}
    for sample in samples:
        role = sample["role"]
        if role not in role_endpoints:
            role_order.append(role)
            role_endpoints[role] = []
        endpoint = sample["endpoint"]
        if endpoint not in role_endpoints[role]:
            role_endpoints[role].append(endpoint)
    base_names = {"P": "P Prefill", "D": "D Decode", "M": "混部"}
    for role in role_order:
        multi = len(role_endpoints[role]) > 1
        base = base_names.get(role, role_label(role))
        name = f"{base} 平均（{len(role_endpoints[role])} 实例）" if multi else base
        buckets: Dict[datetime, List[float]] = {}
        for sample in samples:
            if sample["role"] != role:
                continue
            value = sample.get("kv_cache_usage_pct")
            if value is None:
                continue
            ts = _to_dt(sample["timestamp"])
            if ts is None:
                continue
            buckets.setdefault(ts.replace(microsecond=0), []).append(float(value))
        if not buckets:
            continue
        xs, ys, hover = [], [], []
        for key in sorted(buckets):
            values = buckets[key]
            avg = sum(values) / len(values)
            suffix = f"（{len(values)} 实例平均）" if multi else ""
            xs.append(key)
            ys.append(avg)
            hover.append(
                f"角色: {name}<br>"
                f"采集时间: {key.strftime('%Y-%m-%d %H:%M:%S')}<br>"
                f"KV Cache 占用率: {_fmt_float(avg, 2, '%')}{suffix}"
            )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=name,
                legend=legend,
                legendgroup=f"{legend}-{role}",
                line=dict(color=ROLE_COLORS.get(role, "#444444"), width=2),
                marker=dict(size=5),
                hovertemplate="%{text}<extra></extra>",
                text=hover,
                connectgaps=False,
            ),
            row=row,
            col=col,
        )
    fig.update_yaxes(title_text="KV Cache 占用率（%）", **AXIS_CONFIG, rangemode="tozero", row=row, col=col)
    fig.update_xaxes(title_text="时间", type="date", tickformat="%H:%M:%S", **AXIS_CONFIG, row=row, col=col)


def _add_queue_bars(fig, samples: List[Dict[str, Any]], role: str, row: int, col: int, legend: str) -> None:
    xs, running, waiting, preempted, hover, show_preempted = bin_queue_series(samples, role)
    label = role_label(role)
    fig.add_trace(
        go.Bar(
            x=xs,
            y=running,
            name=f"{label} Running",
            legend=legend,
            legendgroup=f"{legend}-running",
            marker_color=RUNNING_COLOR,
            width=QUEUE_BAR_WIDTH_MS,
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover,
            showlegend=True,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Bar(
            x=xs,
            y=waiting,
            name=f"{label} Waiting",
            legend=legend,
            legendgroup=f"{legend}-waiting",
            marker_color=WAITING_COLOR,
            width=QUEUE_BAR_WIDTH_MS,
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover,
            showlegend=True,
        ),
        row=row,
        col=col,
    )
    if show_preempted:
        fig.add_trace(
            go.Bar(
                x=xs,
                y=preempted,
                name=f"{label} Running (preempted)",
                legend=legend,
                legendgroup=f"{legend}-preempted",
                marker_color=PREEMPTED_COLOR,
                width=QUEUE_BAR_WIDTH_MS,
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover,
                showlegend=True,
            ),
            row=row,
            col=col,
        )
    y_title = f"{label} 队列请求数（Running+Waiting"
    y_title += "+Preempted）" if show_preempted else "）"
    fig.update_yaxes(
        title_text=y_title,
        **AXIS_CONFIG,
        rangemode="tozero",
        row=row,
        col=col,
    )
    fig.update_xaxes(
        title_text="时间（每1秒一组）",
        type="date",
        tickformat="%H:%M:%S",
        **AXIS_CONFIG,
        row=row,
        col=col,
    )




def _lighten_color(hex_color: str, factor: float) -> str:
    """Lighten a hex color by blending towards white."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"

def _add_kv_lines_per_engine(fig, samples: List[Dict[str, Any]], row: int, col: int, legend: str, role: str = "M") -> None:
    """图5: 每个 DP engine 的 KV Cache 占用率折线图（8条线）。"""
    engine_ids: set = set()
    for sample in samples:
        if sample.get("role") != role:
            continue
        engines = sample.get("engines") or {}
        engine_ids.update(engines.keys())
    if not engine_ids:
        return
    for eid in sorted(engine_ids, key=int):
        xs, ys, hover = [], [], []
        for sample in samples:
            if sample.get("role") != role:
                continue
            engines = sample.get("engines") or {}
            eng = engines.get(eid)
            if not eng:
                continue
            xs.append(_to_dt(sample.get("timestamp")))
            raw = eng.get("kv_cache_usage_raw")
            pct = float(raw) * 100.0 if raw is not None else 0.0
            ys.append(pct)
            hover.append(
                f"DP-{eid}<br>"
                f"时间: {sample.get('timestamp')}<br>"
                f"KV Cache 占用率: {pct:.2f}%"
            )
        color = DP_LINE_PALETTE[int(eid) % len(DP_LINE_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=f"DP-{eid}",
                legend=legend,
                legendgroup=f"{legend}-dp{eid}",
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate="%{text}<extra></extra>",
                text=hover,
                connectgaps=False,
                showlegend=True,
            ),
            row=row,
            col=col,
        )
    fig.update_yaxes(title_text="KV Cache 占用率（%）", **AXIS_CONFIG, rangemode="tozero", row=row, col=col)
    fig.update_xaxes(title_text="时间", type="date", tickformat="%H:%M:%S", **AXIS_CONFIG, row=row, col=col)


def _add_queue_bars_per_engine(fig, samples: List[Dict[str, Any]], role: str, row: int, col: int, legend: str) -> None:
    """图6: 每个 DP engine 的 Running/Waiting 堆叠柱形图（8个DP，每个DP有running+waiting两色）。"""
    engine_ids: set = set()
    for sample in samples:
        if sample.get("role") != role:
            continue
        engines = sample.get("engines") or {}
        engine_ids.update(engines.keys())
    if not engine_ids:
        return
    sorted_eids = sorted(engine_ids, key=int)
    n_engines = len(sorted_eids)
    bar_width_s = 0.08  # 80ms per bar
    # x offset for each engine within the 1-second bin
    offsets = {eid: (int(eid) - (n_engines - 1) / 2.0) * bar_width_s for eid in sorted_eids}

    from datetime import timedelta as _td
    bar_width_ms = bar_width_s * 1000

    for eid in sorted_eids:
        xs, ys_r, ys_w, hover = [], [], [], []
        for sample in samples:
            if sample.get("role") != role:
                continue
            engines = sample.get("engines") or {}
            eng = engines.get(eid)
            if not eng:
                continue
            ts = _to_dt(sample.get("timestamp"))
            if ts is None:
                continue
            ts = ts + _td(seconds=offsets[eid])
            run_v = float(eng.get("num_running") or 0)
            wait_v = float(eng.get("num_waiting") or 0)
            xs.append(ts)
            ys_r.append(run_v)
            ys_w.append(wait_v)
            hover.append(
                f"DP-{eid}<br>"
                f"时间: {sample.get('timestamp')}<br>"
                f"Running: {int(run_v)}<br>"
                f"Waiting: {int(wait_v)}"
            )
        color = DP_LINE_PALETTE[int(eid) % len(DP_LINE_PALETTE)]
        # Lighter shade for waiting (same hue, higher lightness)
        waiting_color = _lighten_color(color, 0.45)
        # Running bar (bottom of stack, shows in legend)
        fig.add_trace(
            go.Bar(
                x=xs,
                y=ys_r,
                name=f"DP-{eid}",
                legend=legend,
                legendgroup=f"{legend}-dp{eid}",
                marker_color=color,
                width=bar_width_ms,
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover,
                showlegend=True,
            ),
            row=row,
            col=col,
        )
        # Waiting bar (top of stack, lighter shade, no legend)
        fig.add_trace(
            go.Bar(
                x=xs,
                y=ys_w,
                name=f"DP-{eid} Waiting",
                legend=legend,
                legendgroup=f"{legend}-dp{eid}",
                marker_color=waiting_color,
                width=bar_width_ms,
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover,
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    fig.update_yaxes(title_text="各 DP 队列请求数", **AXIS_CONFIG, rangemode="tozero", row=row, col=col)
    fig.update_xaxes(title_text="时间（每1秒一组）", type="date", tickformat="%H:%M:%S", **AXIS_CONFIG, row=row, col=col)

def _add_phase_bands(fig, phases: List[Dict[str, str]], stopped_at: Optional[datetime], cells: List[Tuple[int, int]]) -> None:
    phase_colors = ["rgba(31,119,180,0.08)", "rgba(255,127,14,0.08)", "rgba(44,160,44,0.08)"]
    for idx, phase in enumerate(phases):
        start = _to_dt(phase.get("start"))
        end = _to_dt(phase.get("end")) or stopped_at or datetime.now()
        if start is None:
            continue
        color = phase_colors[idx % len(phase_colors)]
        for row, col in cells:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=color,
                line_width=0,
                layer="below",
                row=row,
                col=col,
            )
            fig.add_vline(
                x=start,
                line_dash="dot",
                line_color="#666666",
                line_width=1,
                row=row,
                col=col,
            )


def append_hit_rate_to_aisbench_json(perf_dir: str, summary: List[Dict[str, Any]]) -> None:
    if not perf_dir or not os.path.isdir(perf_dir) or not summary:
        return
    for name in os.listdir(perf_dir):
        if not name.endswith(".json"):
            continue
        low = name.lower()
        if "rps_distribution" in low or "vllm_pd" in low:
            continue
        path = os.path.join(perf_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        metric = data.setdefault("Prefix Cache Hit Rate", {})
        if not isinstance(metric, dict):
            continue
        for item in summary:
            rate = item.get("hit_rate")
            metric[hit_rate_stage_label(item, summary)] = f"{rate:.2f}%" if rate is not None else "N/A"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)


def write_metrics_html(
    output_path: str,
    samples: List[Dict[str, Any]],
    phases: List[Dict[str, str]],
    started_at: Optional[datetime] = None,
    stopped_at: Optional[datetime] = None,
    baseline_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    aisbench_plot_path: Optional[str] = None,
    aisbench_plots: Optional[List[Dict[str, Any]]] = None,
    hit_rate_summary: Optional[List[Dict[str, Any]]] = None,
    aisbench_log_path: str = "aisbench.log",
    mode: str = "pd",
) -> None:
    if go is None or make_subplots is None:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("<html><body><h3>未安装 plotly，无法生成指标图。请 pip install plotly</h3></body></html>")
        return

    colocated = mode == "colocated"
    if colocated:
        titles = [
            "图1 请求时间轴（来自 gsm8k_plot）",
            "图2 KV Cache 占用率随时间变化",
            "图3 请求并发数（来自 gsm8k_plot）",
            "图4 混部 Running/Waiting/Preempted 队列（每1秒）",
            "图5 各 DP KV Cache 占用率",
            "图6 各 DP Running/Waiting 队列",
            "图7 ais_bench 测试结果",
        ]
        specs = [[{}, {}], [{}, {}], [{}, {}], [{"type": "table", "colspan": 2}, None]]
        cartesian_cells = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
        legend_keys = ["legend", "legend2", "legend3", "legend4", "legend5", "legend6"]
        prefers = ["topright", "topleft", "topright", "topright", "topright", "topright"]
        page_title = "vLLM 混部运行时指标与 ais_bench 请求图"
        n_rows = 4
        table_row = 4
        table_col = 1
    else:
        n_rows = 5
        table_row = 5
        table_col = 2
        titles = [
            "图1 请求时间轴（来自 gsm8k_plot）",
            "图2 KV Cache 占用率随时间变化",
            "图3 请求并发数（来自 gsm8k_plot）",
            "图4 P 节点 Running/Waiting/Preempted 队列（每1秒）",
            "图5 P 节点各 DP KV Cache 占用率",
            "图6 D 节点 Running/Waiting/Preempted 队列（每1秒）",
            "图7 P 节点各 DP Running/Waiting 队列",
            "图8 D 节点各 DP KV Cache 占用率",
            "图9 D 节点各 DP Running/Waiting 队列",
            "图10 ais_bench 测试结果",
        ]
        specs = [[{}, {}], [{}, {}], [{}, {}], [{}, {}], [{}, {"type": "table"}]]
        cartesian_cells = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1))
        legend_keys = ["legend", "legend2", "legend3", "legend4", "legend5", "legend6", "legend7", "legend8", "legend9"]
        prefers = ["topright", "topleft", "topright", "topright", "topright", "topright", "topright", "topright", "topright"]
        page_title = "vLLM PD 分离运行时指标与 ais_bench 请求图"

    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=titles,
        vertical_spacing=0.11,
        horizontal_spacing=0.08,
        specs=specs,
    )

    origin = infer_plot_origin(samples, phases, started_at, aisbench_log_path, aisbench_plot_path)
    plot_layout = {}
    plot_specs = [dict(item) for item in (aisbench_plots or []) if item.get("path") and os.path.exists(item["path"])]
    if not plot_specs and aisbench_plot_path and os.path.exists(aisbench_plot_path):
        plot_specs = [{"path": aisbench_plot_path, "label": "", "origin": origin}]
    multi_plot = len(plot_specs) > 1
    for spec_idx, rec in enumerate(plot_specs):
        path = rec["path"]
        try:
            spec = load_plotly_spec_from_html(path)
            if not spec:
                logging.warning("Failed to parse gsm8k_plot.html: %s", path)
                continue
            first_traces, second_traces, layout = split_aisbench_plot_traces(spec)
            if spec_idx == len(plot_specs) - 1 or not plot_layout:
                plot_layout = layout
            rec_origin = rec.get("origin")
            if isinstance(rec_origin, str):
                rec_origin = _to_dt(rec_origin)
            if rec_origin is None:
                rec_origin = infer_plot_origin(samples, phases, started_at, None, path)
            label = str(rec.get("label") or "").strip()
            prefix = f"{label} " if multi_plot and label else ""
            dash = "dot" if multi_plot and spec_idx < len(plot_specs) - 1 else None
            _attach_copied_traces(fig, first_traces, 1, 1, "legend", rec_origin, prefix, dash)
            _attach_copied_traces(fig, second_traces, 2, 1, "legend3", rec_origin, prefix, dash)
            logging.info(
                "embedded gsm8k_plot traces: label=%s timeline=%s concurrency=%s from %s",
                label or "-",
                len(first_traces),
                len(second_traces),
                path,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to embed gsm8k_plot.html %s: %s", path, exc)

    y1_title = _layout_axis_title(plot_layout, "yaxis", "Request Index")
    if "Relative" in y1_title:
        y1_title = "Request Index"
    y3_title = _layout_axis_title(plot_layout, "yaxis2", "Request Concurrency Count")
    fig.update_yaxes(title_text=y1_title, **AXIS_CONFIG, row=1, col=1)
    fig.update_yaxes(title_text=y3_title, **AXIS_CONFIG, row=2, col=1)
    fig.update_xaxes(title_text="时间", type="date", tickformat="%H:%M:%S", **AXIS_CONFIG, row=1, col=1)
    fig.update_xaxes(title_text="时间", type="date", tickformat="%H:%M:%S", **AXIS_CONFIG, row=2, col=1)

    _add_kv_lines(fig, samples, row=1, col=2, legend="legend2")
    if colocated:
        _add_queue_bars(fig, samples, role="M", row=2, col=2, legend="legend4")
        _add_kv_lines_per_engine(fig, samples, row=3, col=1, legend="legend5", role="M")
        _add_queue_bars_per_engine(fig, samples, role="M", row=3, col=2, legend="legend6")
    else:
        _add_queue_bars(fig, samples, role="P", row=2, col=2, legend="legend4")
        _add_queue_bars(fig, samples, role="D", row=3, col=2, legend="legend6")
        _add_kv_lines_per_engine(fig, samples, row=3, col=1, legend="legend5", role="P")
        _add_kv_lines_per_engine(fig, samples, row=4, col=2, legend="legend8", role="D")
        _add_queue_bars_per_engine(fig, samples, role="P", row=4, col=1, legend="legend7")
        _add_queue_bars_per_engine(fig, samples, role="D", row=5, col=1, legend="legend9")
    _add_phase_bands(fig, phases, stopped_at, list(cartesian_cells))

    perf_dir = os.path.dirname(os.path.abspath(output_path))
    csv_rows, json_rows = load_aisbench_result_tables(perf_dir)
    fig.add_trace(_build_result_table_trace(csv_rows, json_rows), row=table_row, col=table_col)

    for row, col in cartesian_cells[1:]:
        fig.update_xaxes(matches="x", type="date", tickformat="%H:%M:%S", row=row, col=col)
    fig.update_xaxes(type="date", tickformat="%H:%M:%S", row=1, col=1)

    axis_pairs = _ordered_cartesian_axis_pairs(fig)
    legend_layout = {}
    for (_row, _col), (xk, yk), legend_key, prefer in zip(
        cartesian_cells, axis_pairs, legend_keys, prefers
    ):
        short_x = "x" if xk == "xaxis" else "x" + xk.replace("xaxis", "")
        ys = []
        for trace in fig.data:
            if getattr(trace, "type", None) == "table":
                continue
            if str(getattr(trace, "xaxis", "x") or "x") == short_x and getattr(trace, "y", None) is not None:
                ys.extend(list(trace.y))
        legend_layout[legend_key] = _adaptive_legend_box(fig, xk, yk, ys, prefer)

    hit_line = format_hit_rate_parenthetical(hit_rate_summary)
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{page_title}</b>"
                f"<br><span style='font-size:14px;font-weight:400'>{hit_line}</span>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=22, color="#1a1a1a"),
        ),
        height=1950 if colocated else 2400,
        width=1680,
        plot_bgcolor="white",
        paper_bgcolor="white",
        barmode="stack",
        bargap=0.15,
        hovermode="closest",
        dragmode="zoom",
        margin=dict(l=70, r=36, t=92, b=48),
        **legend_layout,
    )
    for anno in fig.layout.annotations or []:
        text = str(anno.text or "")
        if text.startswith("图"):
            anno.font = dict(size=13, color="#222222")

    fig.write_html(
        output_path,
        include_plotlyjs=True,
        config=WEBGL_CONFIG,
        auto_open=False,
        full_html=True,
        post_script=RELAYOUT_SYNC_JS,
    )
