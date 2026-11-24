# PD 请求五阶段耗时分析：实现说明

> 配套：《vLLM-Ascend PD 分离 KV 传输 Metrics 上报开发指南》（v2，2026-10-16）
> 代码：`kv_stage_analyzer.py`（解析/聚合）+ `vllm_pd_monitor.py`（出图）+ `aisbench_test.py`（调度）
> 状态：已实现并随工具发布（本文档与实现同步）

## 1. 目标

vLLM（v2 方案）在 PD 分离部署中提供两级观测数据：**REQ_STAGE_METRIC**（每请求五阶段耗时，P/D 各一条日志）与 **KV_XFER_METRIC**（D 侧每 worker 传输明细）。本功能在工具侧消费这些日志，做到：

| 目标 | 说明 |
| --- | --- |
| 自动统计 | 测试结束自动解析全量阶段，输出五阶段均值/P90/max 与总和占比 |
| 可视化 | 在 runtime metrics HTML 中新增**图 3「请求五阶段耗时构成」**：每请求一条水平分段时间线（P 等待 / P 执行 / KV 拉取 / D 等待 / D 执行） |
| 诊断增强 | 将 `kv_transfer_s` 分解为「纯传输」与「等待/建连/失败超时」 |
| 易配置 | 仅新增 2 个配置项，留空即禁用；对现有功能零影响 |
| 口径一致 | 统计仅含 full_dataset 阶段（warmup 排除），CSV 明细保留 warmup |

## 2. 数据源

| 数据 | 粒度 | 格式 | 来源 |
| --- | --- | --- | --- |
| REQ_STAGE_METRIC（P） | 每请求 1 条 | `REQ_STAGE_METRIC request_id=<id> p_waiting_s=<f> p_running_s=<f> ts_ms=<epoch_ms>` | vllm_p.log |
| REQ_STAGE_METRIC（D） | 每请求 1 条 | `REQ_STAGE_METRIC request_id=<id> kv_transfer_s=<f> d_waiting_s=<f> d_running_s=<f> ts_ms=<epoch_ms>` | vllm_d.log |
| KV_XFER_METRIC | 每请求 × 每 worker | `KV_XFER_METRIC request_id=<id>-<会话后缀> elapsed_ms=<f> bytes=<int> local_blocks=<int> local_ip=<ip> local_device_id=<int>` | vllm_d.log |
| KV_XFER_FAIL | 全局 | `KV_XFER_FAIL ret=-1, continuing`（无 request_id） | vllm_d.log |

字段语义：`p_waiting_s`=P 排队（QUEUED→SCHEDULED）；`p_running_s`=P prefill（SCHEDULED→首 token）；`kv_transfer_s`=D 侧 KV 等待+传输（KV_LOAD_STARTED→KV_LOAD_FINISHED）；`d_waiting_s`=D 纯排队（已扣除 KV 等待）；`d_running_s`=D 推理（SCHEDULED→末 token）；`ts_ms`=日志发出时刻（毫秒），用作时间线锚点。

## 3. 数据流

```text
run_aisbench 前 : KVStageAnalyzer.start()          # 记录 P/D 日志当前字节 offset
ais_bench 压测中: （无干预）
run_aisbench 后 : KVStageAnalyzer.collect(phase)    # 增量读取至最后一个完整换行
    ├─ 解析 REQ_STAGE_METRIC → P 行 / D 行
    ├─ 解析 KV_XFER_METRIC  → request_id 去会话后缀后按请求聚合 worker 明细
    ├─ 解析 KV_XFER_FAIL    → 全局失败计数
    └─ join：以 D 行为主序生成本轮记录；加入批内去重；P 缺失记录登记 pending
测试结束        : summary → log_summary + aisbench_result.csv 5 列
                : align_with_aisbench_details + write_details_csv
                : build_bar_traces() → write_metrics_html()（图 3）
```

关键决策：

- **offset 增量采集**：`start()` 记录文件大小，`collect()` 只读新增段；采集按**二进制**读取，且 offset 只推进到最后一个完整换行，末尾半行留待下次——避免文本模式按字节 seek 的未定义行为与「半行被永久跳过」
- **模块化**：`kv_stage_analyzer.py` 负责解析/聚合/trace，`vllm_pd_monitor.py` 只扩展布局与注入点，`aisbench_test.py` 只加 start/collect 两处调用

## 4. 配置

```python
KV_STAGE_LOG_P = "/workspace/pd/vllm_p.log"   # P 节点日志
KV_STAGE_LOG_D = "/workspace/pd/vllm_d.log"   # D 节点日志
```

| 设计点 | 行为 |
| --- | --- |
| 留空禁用 | 两项任一为空 → 不启用；**只填一项会打 warning 说明需同时填写** |
| 混部禁用 | 混部形态（`M_LISTEN_SERVER`）配置后由 `aisbench_test.py` 告警并忽略 |
| 文件缺失 | `start()` 时路径不存在 → 自动降级禁用 + warning，不中断测试 |
| 补丁缺失 | 有新增日志却解析到 0 条 `REQ_STAGE_METRIC` → 一次性 warning，提示 vLLM 侧需携带 PR #62685 |
| 跨机 | 路径是运行 aisbench 的机器上的本地路径；跨机需挂载或拷贝 D 侧日志 |

## 5. 解析规则

### 5.1 正则

```python
_REQ_STAGE_RE = re.compile(r"REQ_STAGE_METRIC\s+request_id=(\S+)\s+(.*)")   # 键值对逐个提取
_FIELD_RE     = re.compile(r"(\w+)=(\S+)")
_TS_MS_RE     = re.compile(r"\bts_ms=(\d+(?:\.\d+)?)")                      # 优先毫秒锚点
_KV_XFER_RE   = re.compile(
    r"KV_XFER_METRIC\s+request_id=(\S+)\s+elapsed_ms=(\S+)\s+bytes=(\d+)"
    r"\s+local_blocks=(\d+)\s+local_ip=(\S+)\s+local_device_id=(\d+)"
)                                                                          # 组号 1..6
_KV_XFER_FAIL_RE = re.compile(r"KV_XFER_FAIL")
```

`KV_XFER_METRIC` 的 request_id 形如 `chatcmpl-<uuid4>-<传输会话后缀>`，用 `rsplit("-", 1)[0]` 去掉最后一段得到与 P/D 一致的 request_id（不能用「首个 `-` 切分」，uuid 本身含 `-`）。

### 5.2 join 与聚合

- **join 以 D 行为主序**（D 行代表请求完成）。同一 `request_id` 的 P 行在本轮或后续轮到达都可合并
- **批内去重**：本轮新增记录后立即加入已知集合，重复的 D 行不会生成两条记录
- **迟到 P 行回填**：D 先到、P 未到的记录进入 pending；后续 `collect()` 若读到其 P 行，原地补齐 `p_waiting/p_running/t_p_end` 并置 `missing_p=False`（保留原 `phase`/`seq`）
- **P 有 D 无**（请求未完成/失败）不生成记录，仅计数告警
- **worker 聚合**：`worker_total`/`worker_success`/`kv_bytes_total`/`elapsed_ms_min/max/avg`、`worker_device_ids`
- **失败 worker 推断**：`KV_XFER_FAIL` 无 request_id，用全局成功条数众数作为期望 worker 数，请求级 `worker_failed ≈ 期望数 − worker_success`（图上以 "≈" 标注）

### 5.3 KV 等待分解

```text
kv_wait_overhead_s = kv_transfer_s − max(elapsed_ms)/1000     # 下限 0
```

含义：调度器视角 KV 耗时中扣除最慢 worker 纯传输后剩余部分 = 建连 + 排队 + 失败 group 超时。该分解假设各 worker 并行传输；串行/多轮场景下为近似值。

### 5.4 边界与容错

| 场景 | 行为 |
| --- | --- |
| 日志轮转（offset > 文件大小） | 回退为全量重读 + warning（可能混入历史请求） |
| 最后一行未写完 | 只消费到最后一个 `\n`，半行保留到下次读取 |
| 格式不匹配行 | 跳过并计数；首次异常行打 warning（含样本，截断 200 字符） |
| 请求量 > 2000 | 图表均匀降采样至 2000 条；**统计与 CSV 始终全量** |
| 单侧缺失 | P 缺失记录仍保留 D 侧数据：P 阶段统计只用 joined 记录，D/KV 阶段统计用全部 D 记录 |

## 6. 图表

### 6.1 图 3：请求五阶段耗时构成

- **类型**：水平分段时间线（Gantt 风格 bar），X 轴为绝对时间（datetime）
- **时间锚点**：优先 `ts_ms`；缺失时退回日志行首时间戳（秒级），并把 KV 起点钳制到 P 结束点以免段覆盖（仅影响显示，不改统计值）
- **Y 轴**：请求编号。全量请求与图 1 的 Request Index 同号（对齐 `details.jsonl` 后为 `client_id+1`）；warmup 用独立编号以半透明叠加
- **堆叠顺序**：p_waiting → p_running → kv_transfer → d_waiting → d_running
- **配色**：p_waiting `#94a3b8` / p_running `#3b82f6` / kv_transfer `#f97316` / d_waiting `#cbd5e1` / d_running `#22c55e`
- **图例**：位于图正下方，可点击切换显隐，名称附各阶段 max/min/avg
- **hover**：request_id 后 8 位、五阶段耗时、总时延、KV 分解（纯传输 max/avg、等待开销、worker 成功/失败(≈)、KV 数据量 MB）

### 6.2 汇总统计

打屏 + `aisbench_result.csv` 追加 5 列均值。统计项：每阶段 avg / P90 / max / 总和占比；KV 分解额外给出纯传输与等待开销的 avg / P90 / max（不计算 P50）。

## 7. 产物

| 产物 | 位置 | 内容 |
| --- | --- | --- |
| 图 3 | `vllm_pd_runtime_metrics.html` | 五阶段水平分段时间线 + hover 明细 |
| 汇总 5 列 | `aisbench_result.csv` | `p_waiting_avg_s` / `p_running_avg_s` / `kv_transfer_avg_s` / `d_waiting_avg_s` / `d_running_avg_s` |
| 请求明细 | `performances/vllm-api-stream-chat/kv_stage_details.csv` | 每请求：五阶段值、worker 成功/失败、kv_bytes_total、elapsed_ms_max、kv_wait_overhead_s、客户端跨度、对齐后 Request Index（含 warmup 行） |
| 打屏汇总 | `aisbench.log` | §6.2 统计表 |

## 8. 风险与已知限制

| 项 | 说明 | 对策 |
| --- | --- | --- |
| 依赖未合入的上游 PR | `REQ_STAGE_METRIC` 来自 draft PR #62685，格式仍可能变化 | 补丁缺失时显式告警；解析为键值对，字段顺序不敏感 |
| kv_transfer 与 d_waiting 的口径 | 依赖 vLLM 侧 `pure_queue_time` 已扣除 KV 等待，否则会重复计时 | 工具不做二次校验；以 host 侧口径为准 |
| KV_XFER_FAIL 无 request_id | 失败 worker 无法精确归属 | 众数启发式 + "≈" 标注 |
| 线性轴量级差异 | kv 20s 与 d_waiting 3ms 同图时小阶段不可见 | 汇总统计补充；后续可加箱线图（log 轴） |
| 多轮 `--repeat` | 仅最后一轮参与 `details.jsonl` 对齐，早期轮次用独立编号，可能与对齐编号重叠 | 需要多轮对齐时分开跑 |
| 与 stable_stage 的关系 | 统计全部 full_dataset 请求（default_perf 口径） | 定位为工程诊断图，不做稳态子集过滤 |
