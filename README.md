# aisbench-prefix-test-tool

基于 [ais_bench](https://github.com/AISBench/benchmark) 的 vLLM 推理服务测试工具：**性能测试**、**Prefix Cache 命中率测试**、**运行时指标监控可视化**（混部 / PD 分离）。

> 项目来源：参考 [rayn-zzz/aisbench_auto_tools_prefix](https://github.com/rayn-zzz/aisbench_auto_tools_prefix) 部分内容重新设计开发。

## 1. 功能

- **性能测试**：定长/不定长输入、流式/非流式、思考模式、稳态测试
- **Prefix Cache 测试**：自动生成带公共前缀的数据集，warmup 预热后跑全量，统计各 DP 域命中率
- **运行时监控**：测试期间每秒采集 vLLM `/metrics`，生成单文件交互式 HTML（KV 占用、各 DP 队列、请求时间轴，各图独立缩放，图例可点击切换显隐并附 max/min/avg 统计）
- **数据集生成**：基于 GSM8K，可指定长度、前缀重复率、前缀个数
- **精度测试**：GSM8K 格式数据集精度评估

## 2. 快速开始

```bash
# 进入带 ais_bench 的环境（mindie/vllm 镜像均已打包），另需安装：
pip install transformers plotly tabulate requests

git clone https://github.com/gaoyuan777/aisbench-prefix-test-tool.git
cd aisbench-prefix-test-tool

# 按部署形态复制对应模板为 config.py，修改其中的地址/路径
cp config_template_colocated.py config.py   # 混部
# 或 cp config_template_pd.py config.py     # PD 分离

python3 aisbench_test.py --input_len 2048 --output_len 2048 --data_num 160 --concurrency 40 --request_rate 10
```

`DATASET_PATH` 指向的文件夹需提前创建。

## 3. config.py 模板（二选一）

| 形态 | 模板文件 | 关键差异 |
| --- | --- | --- |
| 混部 | `config_template_colocated.py` | `HOST_*` 填 vLLM 服务地址；监控只填 `M_LISTEN_SERVER`；`POD_INFO` 填服务地址 |
| PD 分离 | `config_template_pd.py` | `HOST_*` 填 **proxy** 地址；`POD_INFO` 填 **P 节点各 DP 域端口**；监控填 `P_LISTEN_SERVER` / `D_LISTEN_SERVER` |

公共规则：

- `M_LISTEN_SERVER` 优先级最高，非空时忽略 P/D 监听；三个监听全空则不启用运行时监控
- `*_LISTEN_SERVER` 必须是 vLLM 引擎自身的 metrics 端口（多 DP 域逗号分隔），**不要填 proxy**（其 `/metrics` 常为 404）
- `POD_INFO` 只服务 `--prefix_test` 的命中率统计，与运行时监控无关；留空 `[]` 默认 `HOST_IP:HOST_PORT`

## 4. DP 部署形态与 metrics 配置

vLLM 的 DP 部署有三种形态，`/metrics` 暴露方式不同，直接决定 `POD_INFO` 与 `*_LISTEN_SERVER` 的填法：

以 dp=4、服务 IP `192.168.1.10`、起始端口 9000 为例：

| 形态 | 部署方式 | metrics 端口 | 每端口返回 |
| --- | --- | --- | --- |
| A. 单实例内部 DP | `vllm serve --data-parallel-size 4 --api-server-count 1` | 1 个 | 全部 4 个 DP 的数据（prometheus `engine="0"~"3"` 标签区分） |
| B. 多 API server | 同上但 `--api-server-count 4`，端口递增 | 4 个 | 每端口仅本 DP 域 |
| C. 多实例外部 DP | `launch_online_dp.py`，每 DP 一个独立 `vllm serve` 进程，`--vllm-start-port` 递增 | 4 个 | 每端口仅本实例 |

**配置用例**（dp=4、服务 IP `192.168.1.10`、起始端口 9000）：

**形态 A**（填 1 个地址即可）：

```python
M_LISTEN_SERVER = "192.168.1.10:9000"
POD_INFO = ["192.168.1.10:9000"]
```

**形态 B / C**（列出全部端口）：

```python
M_LISTEN_SERVER = "192.168.1.10:9000,192.168.1.10:9001,192.168.1.10:9002,192.168.1.10:9003"
POD_INFO = [
    "192.168.1.10:9000", "192.168.1.10:9001",
    "192.168.1.10:9002", "192.168.1.10:9003",
]
```

PD 分离场景将 `M_LISTEN_SERVER` 替换为 `P_LISTEN_SERVER` / `D_LISTEN_SERVER`，填法同理。

服务运行时一条命令判定形态：

```bash
curl -s http://{ip}:{port}/metrics | grep -o 'engine="[0-9]*"' | sort -u
# 返回 dp 数行（engine="0"~"3"）→ 形态 A，填 1 个地址；返回 1 行或空 → 形态 B/C，列全部端口
```

两种展开方式（1 地址 × N 个 engine 标签 / N 地址 × 各 1 个 engine）对监控等价：per-DP 图表与聚合平均线均正确，只需保证地址列表覆盖全部 DP 域。

## 5. 命令行参数

`python3 aisbench_test.py --help` 查看全部。

| 参数 | 释义 |
| --- | --- |
| `--input_len` / `--output_len` | 输入 / 输出长度（两者之和不可超过服务 `max_model_len`） |
| `--data_num` | 数据集条数 |
| `--concurrency` | 系统最大并发数 |
| `--request_rate` | 请求频率，默认 0（不限流） |
| `--test_type` | `stream`（默认）/ `text` |
| `--dataset` | 指定测试数据集路径（gsm8k 格式） |
| `--repeat` | 单条命令重复测试次数，默认 1 |
| `--enable_think` | 开启思考模式（DeepSeek V3.1 等），默认 false |
| `--test_accuracy` | 精度测试（仅 gsm8k 数据集），默认 false |
| `--npu_num` | NPU 卡数（计算单卡吞吐），默认 1 |
| `--dataset_type` | `normal`（默认）/ `prefix_cache` |
| `--prefix_num` | 前缀种类数，默认 **1** |
| `--repeat_rate` | 前缀重复率，默认 0.5，支持 `"50%"` 或 `"0.5"` |
| `--prefix_test` | 全量测试前先 warmup 预热，默认 false |
| `--seed` | 随机种子（控制随机 token），默认 1 |
| `--dp` | DP 域数量，默认 1 |
| `--output_dir` | 覆盖 config 的 `OUTPUT_DIR` |
| `--length_mean/std/min/max` | 不定长数据集的长度分布参数 |

## 6. Prefix Cache 测试逻辑

**数据集构造**：每条数据 = 公共前缀（`input_len × repeat_rate`）+ 3 个随机 token + 独立后缀；`prefix_num` 个不同前缀轮流挂载。

**warmup 预热**（`--prefix_test`）：

- 发送 **`prefix_num × 2 × dp`** 条纯前缀请求（每前缀每 DP 域 2 条），并发 = `dp`，`output_len=1`，不限流
- 目的：让每个 DP 域的 KV Cache 都存入全部前缀，保证全量阶段稳定命中
- 命中率只统计全量阶段（warmup 排除），通过 `POD_INFO` 各地址的 `prefix_cache_hits/queries` 差值计算

```text
# repeat_rate=0.5、prefix_num=2 的数据集示例
abc123   ← 前缀 abc
def456   ← 前缀 def
abc789   ← 前缀 abc
...
```

## 7. 性能统计口径（stable_stage / default_perf）

config.py 的 `DEFAULT_PERFORMANCE_TEST` 控制 ais_bench 性能统计口径，**默认 `stable_stage`（稳态）**：

| 口径 | 统计范围 | 适用场景 |
| --- | --- | --- |
| `stable_stage` | 仅"稳态窗口"内的请求（并发打满后第 2 个请求 → 最后一次并发峰值） | 排除爬坡/收尾干扰，看服务稳态能力 |
| `default_perf` | 全部请求 | 全生命周期口径 |

- **warmup 阶段固定用 `default_perf`**：warmup 仅 `prefix_num × 2 × dp` 条请求且 `output_len=1`，无法形成稳态窗口，若用 stable_stage 会报 `Can not find a stable stage` 中止测试
- 两种口径的发压流程、输出文件完全一致，仅数值统计范围不同（`Benchmark Duration`/`Total Requests`/`Concurrency` 等在 stable_stage 下为窗口内口径）
- **`aisbench_result.csv` 的 `summarizer` 列**记录每行数据的统计口径，跨口径数据不可直接对比
- 稳态窗口判定失败（数据量太小、并发未两次打满）时测试中止并输出诊断日志；可增大 `--data_num`，或改回 `default_perf`
- 版本注意：ais-bench-benchmark 的稳态算法 2026-08 有大改（窗口判定更鲁棒），PyPI `3.1.20260630` 为旧算法，新旧版本稳态数值不可比；`pip show ais-bench-benchmark` 确认版本

## 8. 使用示例

```bash
# 1. 普通性能测试
python3 aisbench_test.py --input_len 2048 --output_len 2048 --data_num 160 --concurrency 40 --request_rate 10

# 2. 思考模式
python3 aisbench_test.py --input_len 2048 --output_len 2048 --data_num 160 --concurrency 40 --enable_think

# 3. 前缀缓存测试（2 前缀、重复率 50%、dp 2、先预热）
python3 aisbench_test.py --input_len 2048 --output_len 2048 --data_num 160 --concurrency 40 --request_rate 10 \
  --dataset_type prefix_cache --repeat_rate 0.5 --prefix_test --dp 2 --prefix_num 2

# 4. 不预热直接跑全量（测试冷启动命中率）
python3 aisbench_test.py --input_len 2048 --output_len 2048 --data_num 160 --concurrency 40 --request_rate 10 \
  --dataset_type prefix_cache --repeat_rate 73% --prefix_num 3

# 5. 不定长 8k~128k（均值 32k）
python3 aisbench_test.py --input_len 32768 --output_len 300 --data_num 32 --concurrency 8 --request_rate 0 \
  --dataset_type prefix_cache --repeat_rate 90% --prefix_test --dp 2 \
  --length_mean 32768 --length_std 49152 --length_min 8192 --length_max 131072

# 6. 指定数据集 / 精度测试
python3 aisbench_test.py --dataset "/mnt/data/medium.jsonl" --output_len 20 --concurrency 1024
python3 aisbench_test.py --dataset "/mnt/data/gsm8k.jsonl" --output_len 1024 --concurrency 64 --request_rate 4 --test_accuracy
```

## 9. 运行时指标监控

配置 `*_LISTEN_SERVER` 后，测试结束自动生成：

```text
outputs/<目录>/<时间戳>/performances/vllm-api-stream-chat/vllm_pd_runtime_metrics.html
```

- 实时数据覆盖写入 `pd_metrics_live.txt`，另开终端 `tail -F pd_metrics_live.txt` 查看
- 前缀命中率写入 `aisbench_result.csv` 与 HTML 标题下方

**混部布局（7 图）**：请求时间轴 / KV 占用率（聚合平均）/ 并发数 / 队列柱状图 / 各 DP KV / 各 DP 队列 / 结果表

**PD 分离布局（默认 10 图）**：请求时间轴 / KV 占用率（**P/D 各一条平均线**）/ 并发数 / P 队列 / P 各 DP KV / D 队列 / P 各 DP 队列 / D 各 DP KV / D 各 DP 队列 / 结果表

配置 `KV_STAGE_LOG_P/D` 后布局调整为 **11 图 6 行**（行 3 起左 P 右 D 成对），新增**图 3「请求五阶段耗时构成」**水平分段时间线——功能详情、vLLM 依赖补丁与配置用例见[第 10 章](#10-五阶段耗时分析可选)。

各时间轴图表独立缩放、互不影响；图例位于各图正下方（点击可切换数据线显隐，名称附 max/min/avg 统计），底部附参数配置表。

## 10. 五阶段耗时分析（可选）

PD 分离形态下，按请求生命周期拆解 **P 等待 / P 执行 / KV 拉取 / D 等待 / D 执行** 五个阶段的耗时：生成 HTML 图 3 水平分段时间线（全量请求与图 1 Request Index 同号 1..N，warmup 半透明叠加独立编号；时间锚点优先取日志行内 `ts_ms` 毫秒打点，秒级日志锚点自动钳制防段覆盖）与逐请求明细 CSV。与命中率同口径：统计仅含全量阶段（warmup 排除），CSV 明细完整含 warmup。

### 10.1 依赖前提：vLLM 上报补丁

五阶段数据来自 vLLM 的 `REQ_STAGE_METRIC` 结构化日志，由上游 PR 提供：

> [vllm-project/vllm#62685](https://github.com/vllm-project/vllm/pull/62685) — *Add per-request KV load wait and pure queue time metrics*（+87/−1，5 个文件）

**该 PR 为 draft 状态（尚未合入上游 main）**，需在部署环境的 vLLM 中自行携带补丁（cherry-pick 或手工修改 `vllm/v1/metrics/loggers.py`、`vllm/v1/metrics/stats.py`、`vllm/v1/core/sched/scheduler.py`、`vllm/v1/engine/__init__.py`）。补丁内容：

- 两个 engine-core 事件 `KV_LOAD_STARTED` / `KV_LOAD_FINISHED`（scheduler 层测量，与 KVConnectorV1 实现无关，NIXL / Mooncake / SharedStorage 通用）
- 两个 Prometheus 直方图 `vllm:request_kv_wait_time_seconds` / `vllm:request_pure_queue_time_seconds`
- 按 `kv_role` 门控的每请求 `REQ_STAGE_METRIC` 日志（含 `ts_ms` 毫秒时间戳），P/D 实例各自输出；无新增 CLI/环境变量开关

日志格式（每请求完成时输出一行）：

```text
# P 节点（kv_producer）
REQ_STAGE_METRIC request_id=chatcmpl-xxx p_waiting_s=0.000 p_running_s=0.310 ts_ms=1792393063738.905
# D 节点（kv_consumer）
REQ_STAGE_METRIC request_id=chatcmpl-xxx kv_transfer_s=0.989 d_waiting_s=0.002 d_running_s=2.432 ts_ms=1792393064712.310
```

字段语义：

| 字段 | 取值来源 | 含义 |
| --- | --- | --- |
| `p_waiting_s` | `queued_time` | P 节点排队时间 |
| `p_running_s` | `prefill_time` | P 节点 prefill 执行时间 |
| `kv_transfer_s` | `kv_wait_time` | `WAITING_FOR_REMOTE_KVS` 窗口耗时（KV 加载等待） |
| `d_waiting_s` | `pure_queue_time` | D 节点纯排队时间（已扣除 KV 等待） |
| `d_running_s` | `inference_time` | D 节点推理时间 |
| `ts_ms` | `time.time() * 1000` | 日志发出时刻（毫秒），工具的时间线锚点 |

工具另支持解析配套的 `KV_XFER_METRIC` 行（KV 传输 worker 级明细：worker 数、传输字节数、最慢/平均耗时），用于图 3 hover 的「KV 等待分解」（纯传输 vs 建连+排队+失败超时）。该上报来自部署环境的配套改动，**可选**——日志无此行时五阶段主功能不受影响，仅 hover 分解与 CSV 相应列为空。

### 10.2 config 配置用例

在 `config.py` 中追加两项（模板见 `config_template_pd.py`）：

```python
# PD 请求五阶段耗时分析（可选）：
#   依赖 vllm 侧 REQ_STAGE_METRIC 上报（见 10.1 的 PR 说明）
#   P/D 节点 vllm 日志文件路径；留空 "" 则不启用（不生成图 3，其余功能不受影响）
#   单机部署（P/D 同容器）两路径均填本机路径；跨机部署 D 侧日志需挂载/拷贝到本机可访问路径
KV_STAGE_LOG_P = "/workspace/pd/vllm_p.log"   # P 节点日志
KV_STAGE_LOG_D = "/workspace/pd/vllm_d.log"   # D 节点日志
```

注意：

- 路径是**运行 aisbench 的机器**上的本地路径：单机 PD（同容器）直接填容器内日志路径；跨机部署需先把 D 侧日志挂载/同步到本机
- 工具基于日志 offset 增量采集，多轮测试自动衔接，无需清日志
- 混部形态（`M_LISTEN_SERVER`）不支持五阶段分析，配置了也会被忽略并告警

### 10.3 运行与产物

无需额外命令行参数，正常跑测试即可（仅 PD 分离形态生效）。产物：

| 产物 | 说明 |
| --- | --- |
| HTML 图 3 | 每请求一条水平分段时间线（图例可点击切换显隐，名称附 max/min/avg 统计；warmup 半透明叠加） |
| `aisbench_result.csv` | 追加 5 列阶段均值（`p_waiting_avg_s` 等） |
| `kv_stage_details.csv` | 逐请求明细：五阶段耗时、KV 等待分解、worker 传输、客户端跨度、对齐后的 Request Index（含 warmup 行） |

## 11. 结果获取

| 产物 | 位置 |
| --- | --- |
| 性能结果 | `aisbench_result.csv`（`summarizer` 列标记统计口径） |
| 当次日志 | `aisbench.log`（历史：`aisbench_all.log`、`outputs/<目录>/<时间戳>/`） |
| 命中率 | 打屏日志 + `aisbench_result.csv` + HTML 标题 |
| 运行时指标 | 上文 HTML 路径 |
| 五阶段耗时（可选） | HTML 图 3 + `aisbench_result.csv` 5 列均值 + `kv_stage_details.csv` 请求明细 |

## 12. FAQ

**1. 报错"生成数据集失败，请清空 picked ids"**：删除 `picked_ids.txt`。

**2. 命中率不显示/为空**：仅 `--prefix_test` 时统计命中率；检查 `POD_INFO` 配置（混部填服务地址，PD 填 P 节点各 DP 域端口）；手动验证 `unset http_proxy; curl -s http://{ip}:{port}/metrics | grep prefix` 能否返回 `vllm:prefix_cache_*`。

**3. 运行时 HTML 为空/不刷新**：`*_LISTEN_SERVER` 必须是 vLLM 引擎 metrics 端口而非 proxy；混部与 P/D 都填时只走混部；用 `tail -F pd_metrics_live.txt` 看实时数据（每秒覆盖刷新）。

**4. `unrecognized arguments: --num-warmups`**：aisbench 版本过旧，删除 `aisbench_test.py` 中的 `--num-warmups 0`。

**5. tokenizer 加载失败**：检查 transformers 版本与模型适配性，部分模型需 `trust_remote_code=True`。

**6. 报错 "Can not find a stable stage"**：稳态统计未找到稳态窗口（数据量太小或并发未两次打满）。增大 `--data_num`，或将 config.py 的 `DEFAULT_PERFORMANCE_TEST` 改为 `default_perf`。

**7. 五阶段图不生成**：检查 `KV_STAGE_LOG_P/D` 是否填写且日志文件存在（容器内路径）；确认 vllm 侧已合入 REQ_STAGE_METRIC 上报（`grep REQ_STAGE_METRIC vllm_p.log` 有输出）；仅 PD 分离形态支持；`*_LISTEN_SERVER` 未配置时 HTML 不生成（此时仅有打屏汇总与 CSV）。

## 13. License

见 [LICENSE](LICENSE)。

## 14. 致谢

本项目参考了 [rayn-zzz/aisbench_auto_tools_prefix](https://github.com/rayn-zzz/aisbench_auto_tools_prefix) 项目的部分内容，并在此基础上重新设计开发，感谢原作者的工作。
