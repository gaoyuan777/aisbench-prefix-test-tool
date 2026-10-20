# aisbench 工具新功能设计：PD 请求五阶段耗时分析

> 配套文档：《vLLM-Ascend PD分离 KV传输时间Metrics上报开发指南》（v2，2026-10-16）
> 版本：v1 设计稿（2026-10-16）
> 状态：待评审 → 实施

## 1. 背景与目标

vllm 侧（v2 方案）已在 PD 分离部署中提供两级观测数据：**REQ_STAGE_METRIC**（每请求五阶段耗时，P/D 各一条日志）与 **KV_XFER_METRIC**（D 侧每 worker 真实传输明细）。当前这些数据只能靠手工 grep 逐请求分析。

本功能在 aisbench 工具侧完成数据消费，目标：

| 目标 | 说明 |
| --- | --- |
| 自动统计 | 测试结束自动解析全量阶段的 KV 传输时间、P/D 节点内等待与计算时间，无需手工 grep |
| 可视化 | 现有 runtime metrics HTML 新增**图 11「请求五阶段耗时构成」**：每请求一条堆叠柱——P 等待 / P 执行 / KV 拉取 / D 等待 / D 执行 |
| 诊断增强 | 两级日志对照，将 `kv_transfer_s` 分解为「纯传输」与「等待/失败超时」，量化断线卡、传输排队的影响（对应 v2 指南实测案例：22.767s 中 22.6s 为失败超时） |
| 易配置 | 仅新增 **2 个配置项**，留空即禁用；对现有功能零影响 |
| 口径一致 | 与前缀命中率同口径：仅统计 full_dataset 阶段，warmup 自然排除 |

## 2. 数据源（对齐 v2 指南）

| 数据 | 粒度 | 格式 | 来源 |
| --- | --- | --- | --- |
| REQ_STAGE_METRIC（P） | 每请求 1 条 | `REQ_STAGE_METRIC request_id=<id> p_waiting_s=<f> p_running_s=<f>` | vllm_p.log |
| REQ_STAGE_METRIC（D） | 每请求 1 条 | `REQ_STAGE_METRIC request_id=<id> kv_transfer_s=<f> d_waiting_s=<f> d_running_s=<f>` | vllm_d.log |
| KV_XFER_METRIC | 每请求 × 每 worker | `KV_XFER_METRIC request_id=<id>-<uuid> elapsed_ms=<f> bytes=<int> local_blocks=<int> local_ip=<ip> local_device_id=<int> remote_session_id=<ip:port>` | vllm_d.log |
| KV_XFER_FAIL | 全局 | `KV_XFER_FAIL ret=-1, continuing`（无 request_id） | vllm_d.log |

字段定义（引用 v2 指南）：

| 字段 | 定义 |
| --- | --- |
| p_waiting_s | P 排队时间（QUEUED → SCHEDULED） |
| p_running_s | P prefill 执行时间（SCHEDULED → 首 token） |
| kv_transfer_s | D 侧 KV 拉取总时间，等待+传输（KV_LOAD_STARTED → KV_LOAD_FINISHED） |
| d_waiting_s | D 纯排队时间（queued_time − kv_transfer_s） |
| d_running_s | D 全部计算时间（SCHEDULED → 最后 token） |

同一 request_id 贯穿 P/D 日志（proxy 统一生成），据此 join。

## 3. 总体数据流

```text
full_dataset 阶段开始（run_aisbench 前）
  └─ KVStageAnalyzer.start()          # 记录 P/D 日志当前文件 offset（字节）
ais_bench 压测
full_dataset 阶段结束（run_aisbench 后）
  └─ KVStageAnalyzer.collect()        # 从 offset 增量读取至 EOF
      ├─ 解析 REQ_STAGE_METRIC：P 行、D 行分别入库
      ├─ 解析 KV_XFER_METRIC：request_id 去掉 -<uuid> 后缀，按请求聚合 worker 明细
      ├─ 解析 KV_XFER_FAIL：全局失败计数
      └─ 按 request_id join P/D/worker → List[StageRecord]
  ├─ 汇总统计 → logging.info 打屏 + aisbench_result.csv 追加 5 列 + kv_stage_details.csv
  └─ build_traces() → write_metrics_html() 注入图 11
```

关键设计决策：

- **offset 增量采集**：start 时记录文件当前大小，collect 时只读新增部分。天然过滤历史请求与 warmup 阶段，无需时间戳解析、无需 request_id 白名单
- **模块化**：新建独立模块 `kv_stage_analyzer.py`（解析 + 聚合 + trace 构建），`vllm_pd_monitor.py` 仅扩展 `write_metrics_html` 布局与注入点，`aisbench_test.py` 仅加 start/collect 两处调用——与现有 `PDMetricsMonitor` 的接入模式一致

## 4. 配置设计（易配置性）

config.py 新增 2 项（仅 PD 模板需要，混部模板不加）：

```python
# ===== PD 请求五阶段耗时分析（可选）=====
# vllm 侧需已合入 REQ_STAGE_METRIC / KV_XFER_METRIC 上报（见配套开发指南）
# P/D 节点 vllm 日志文件路径，留空 "" 则不启用该功能
# 单机部署（P/D 同容器）：两路径均填本机路径
# 跨机部署：D 侧日志需通过共享存储挂载或测试后拷贝到工具可访问路径
KV_STAGE_LOG_P = "/workspace/pd/vllm_p.log"
KV_STAGE_LOG_D = "/workspace/pd/vllm_d.log"
```

| 设计点 | 说明 |
| --- | --- |
| 留空禁用 | 两项任一为空 → 功能不启用，不生成图 11，其余功能不受影响 |
| 零依赖部署 | 单机部署（v2 指南参考环境：16 卡 P/D 同容器 `/workspace/pd/`）直接填路径即用，无额外步骤 |
| 跨机说明 | 生产跨机形态下 D 日志不在本机：文档指引共享挂载或 scp；V2 预留自动拉取（见 §9） |
| 混部禁用 | colocated 模式下该功能无意义，配置存在时打 warning 并忽略 |
| 文件缺失容错 | start 时文件不存在 → 功能自动降级禁用 + warning，**不中断测试** |

## 5. 解析规则

### 5.1 正则

```python
# REQ_STAGE_METRIC：字段顺序不保证固定，按键值对逐个提取
_REQ_STAGE_RE = re.compile(r"REQ_STAGE_METRIC\s+request_id=(\S+)\s+(.*)")
_FIELD_RE     = re.compile(r"(\w+)=(\S+)")

# KV_XFER_METRIC：request_id 带 -<uuid> 传输会话后缀
_KV_XFER_RE = re.compile(
    r"KV_XFER_METRIC\s+request_id=(\S+?)-(\S+)\s+elapsed_ms=(\S+)\s+bytes=(\d+)"
    r"\s+local_blocks=(\d+)\s+local_ip=(\S+)\s+local_device_id=(\d+)"
)

_KV_XFER_FAIL_RE = re.compile(r"KV_XFER_FAIL")
```

### 5.2 join 与聚合

- **P/D join**：同一 request_id 的 P 行与 D 行合并为一条 `StageRecord`；单侧缺失时保留记录，缺失阶段置 `None`（图上该段高度为 0，hover 标注"P/D 日志缺失"）
- **worker 聚合**：`KV_XFER_METRIC` 的 request_id 去掉 `-<uuid>` 后缀后归属请求，聚合出：
  - `worker_total` / `worker_success`（条数）、`kv_bytes_total`、`elapsed_ms_min/max/avg`
  - `worker_device_ids`（参与传输的卡号列表）
- **失败 worker 推断**：`KV_XFER_FAIL` 行无 request_id 无法精确归属；用启发式——全局成功条数的众数为期望 worker 数，请求级 `worker_failed ≈ 期望数 − worker_success`，hover 中以"≈"标注

### 5.3 KV 等待分解（核心诊断价值）

```text
kv_wait_overhead_s = kv_transfer_s − max(elapsed_ms)/1000
```

- 含义：调度器视角 KV 拉取总时间中，扣除最慢 worker 纯传输后剩余的部分 = 建连 + 排队 + 失败 group 超时等待
- v2 指南实测案例中该项 = 22.767 − 0.146 ≈ 22.6s，即断线卡超时主导
- 呈现：hover 明细 + 汇总统计中独立一行（avg/P90），快速回答"KV 时间花在传输还是等待"

### 5.4 边界与容错

| 场景 | 行为 |
| --- | --- |
| 日志轮转导致 offset > 文件大小 | 回退为全文件解析 + warning（可能混入历史请求，日志中提示） |
| 格式不匹配行 | 跳过并计数，结束时汇总 warning |
| 请求量 > 2000 | 图 11 均匀降采样至 2000 条显示（**统计与 CSV 始终全量**），图标题注明采样 |
| 解析异常 | 单请求级异常不中断整体，计入 error 计数 |

## 6. 图表设计

### 6.1 图 11：请求五阶段耗时构成（V1 主图）

现有 PD 布局（5 行 10 图 + 结果表）新增第 6 行，整行 colspan 2：

```python
# write_metrics_html 中 PD 分支布局变更
specs  = [[{}, {}], [{}, {}], [{}, {}], [{}, {}],
          [{}, {"type": "table"}],
          [{"colspan": 2}, None]]          # ← 新增：图 11 整行
n_rows = 6
titles += ["图11 请求五阶段耗时构成（P 等待 / P 执行 / KV 拉取 / D 等待 / D 执行）"]
```

**图表规格**：

| 项 | 设计 |
| --- | --- |
| 类型 | 纵向 stacked bar（go.Bar × 5，stackgroup 同列） |
| X 轴 | 请求序号（按日志出现顺序即完成顺序），>2000 条降采样 |
| Y 轴 | 秒（linear；stack 不支持 log，量级差异靠统计表与二期箱线图补充） |
| 堆叠顺序（自下而上） | p_waiting → p_running → kv_transfer → d_waiting → d_running（与执行流程一致） |
| 配色 | p_waiting `#94a3b8` 灰 / p_running `#3b82f6` 蓝 / kv_transfer `#f97316` 橙 / d_waiting `#cbd5e1` 浅灰 / d_running `#22c55e` 绿 |
| 图例 | 统一 legend，含各阶段 avg 值（如"KV 拉取 avg=0.42s"） |
| hover | request_id 后 8 位；五阶段精确值；总时延；KV 分解：纯传输 max/avg、等待开销 kv_wait_overhead、worker 成功/失败(≈)数、KV 数据量 MB；缺失侧标注 |

**hover 模板示例**：

```text
req …a4b4ac77 | 总时延 24.71s
P 等待 0.000s | P 执行 1.521s
KV 拉取 22.767s（纯传输 146ms，等待开销 ≈22.62s）
D 等待 0.003s | D 执行 0.368s
worker: 6 成功 / 2 失败(≈) | KV 数据量 76.1MB
```

### 6.2 汇总统计（V1，打屏 + CSV）

测试结束 logging.info 输出，同步落盘：

| 阶段 | avg | P50 | P90 | max | 总和占比 |
| --- | --- | --- | --- | --- | --- |
| P 等待 p_waiting_s | … | | | | …% |
| P 执行 p_running_s | … | | | | …% |
| KV 拉取 kv_transfer_s | … | | | | …% |
| ├─ 纯传输 max(worker elapsed) | … | | | | |
| └─ 等待开销 overhead | … | | | | |
| D 等待 d_waiting_s | … | | | | |
| D 执行 d_running_s | … | | | | |

### 6.3 二期图（V2 预留）

| 图 | 用途 | 说明 |
| --- | --- | --- |
| 五阶段分布箱线图 | 看各阶段量级与离群（log 轴） | 补偿堆叠图线性轴下小阶段不可见的问题 |
| 阶段耗时趋势线 | X=请求完成顺序，5 折线 | 看测试期间各阶段稳定性（前缀命中变化、KV 传输波动） |
| KV worker 散点 | X=device_id，Y=elapsed_ms，点大小=bytes | 定位慢卡/断线卡 |

## 7. 输出物

| 产物 | 位置 | 内容 |
| --- | --- | --- |
| 图 11 | `vllm_pd_runtime_metrics.html` 第 6 行 | 五阶段堆叠图 + hover 诊断明细 |
| 汇总 5 列 | `aisbench_result.csv`（extra_metrics 机制追加） | `p_waiting_avg_s` / `p_running_avg_s` / `kv_transfer_avg_s` / `d_waiting_avg_s` / `d_running_avg_s` |
| 请求明细 | `outputs/<目录>/<时间戳>/performances/vllm-api-stream-chat/kv_stage_details.csv` | 每请求一行：request_id、五阶段值、worker 成功/失败、kv_bytes_total、elapsed_ms_max、kv_wait_overhead_s |
| 打屏汇总 | aisbench.log | §6.2 统计表 + KV 等待分解 |

## 8. 代码改动清单

| 文件 | 改动 | 规模预估 |
| --- | --- | --- |
| `kv_stage_analyzer.py`（新增） | `KVStageAnalyzer` 类：from_config/start/collect/build_traces/summary；正则解析、join、聚合、容错 | ~300 行 |
| `vllm_pd_monitor.py` | `write_metrics_html` 加 `kv_stage_records` 参数；PD 布局 specs/titles 加第 6 行；`_add_kv_stage_bars()` 新函数；export 透传 | ~80 行 |
| `aisbench_test.py` | full_dataset 各 run_aisbench 前后调 start/collect；结果并入 extra_metrics；日志输出汇总 | ~30 行 |
| `config_template_pd.py` | 新增 KV_STAGE_LOG_P/D 配置与注释 | ~8 行 |
| `README.md` | 新章节 + FAQ（日志缺失/跨机获取） | ~40 行 |

## 9. 分期实施

| 版本 | 范围 |
| --- | --- |
| **V1（本次）** | 日志 offset 增量采集 + 解析 join + 图 11 堆叠图 + 汇总打屏 + aisbench_result.csv 5 列 + kv_stage_details.csv + 容错 |
| V2 | ① 跨机自动拉取 D 日志（`KV_STAGE_LOG_D_FETCH = "ssh user@host cat /path"` 自定义命令，输出重定向到临时文件后按本地路径解析）② 三张二期图 ③ 未配日志时的 metrics 直方图差值降级模式（复用现有 /metrics 轮询拉 `vllm:request_kv_wait_time_seconds` 等 4 个直方图的 _sum/_count 差值，给出阶段均值概览，无 per-request 粒度） |

## 10. 风险与开放问题

| 项 | 说明 | 对策 |
| --- | --- | --- |
| KV_XFER_FAIL 无 request_id | 失败 worker 无法精确归属单请求 | 众数启发式估算 + hover "≈" 标注；推动 vllm 侧 v3 给 FAIL 行带上 request_id（建议） |
| 堆叠图线性轴量级差异 | kv 22s vs d_waiting 3ms 同图时小阶段不可见 | 统计表补充；V2 箱线图 log 轴 |
| 日志行无时间戳 | offset 方案已规避时间戳依赖，但无法按时间过滤 | 维持 offset 方案 |
| 超大请求量渲染 | data_num 默认 8192 | 降采样 2000 + 统计全量 |
| 与 stable_stage 口径关系 | 图 11 统计全部 full_dataset 请求（default_perf 口径），非稳态窗口内子集 | 图 11 定位为工程诊断图（全量视角）；如需稳态子集过滤，依赖 aisbench details.jsonl 的 request_id 关联（V3 候选） |
