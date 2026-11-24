# aisbench-prefix-test-tool

基于 [ais_bench](https://github.com/AISBench/benchmark) 的 vLLM 推理服务测试工具，面向昇腾 NPU + PD 分离/混部部署：**性能测试**、**Prefix Cache 命中率测试**、**运行时指标可视化**、**PD 五阶段耗时分析**。

> 参考 [rayn-zzz/aisbench_auto_tools_prefix](https://github.com/rayn-zzz/aisbench_auto_tools_prefix) 重新设计开发。

## 功能

| 功能 | 说明 |
| --- | --- |
| 性能测试 | 定长/不定长输入、流式/非流式、思考模式；`stable_stage`（稳态）/`default_perf`（全量）两种口径 |
| Prefix Cache 测试 | 基于 GSM8K 自动构造"公共前缀 + 3 个唯一 token + 独立后缀"数据集；`--prefix_test` 先 warmup 再跑全量，按 DP 域统计命中率 |
| 运行时监控 | 测试期间每秒采集 vLLM `/metrics`，产出单文件交互式 HTML（KV 占用、各 DP 队列、请求时间轴；图例可点击显隐并附 max/min/avg） |
| 五阶段耗时（可选） | PD 分离下解析 vLLM `REQ_STAGE_METRIC` 日志，输出 P 等待/P 执行/KV 拉取/D 等待/D 执行的水平分段时间线与逐请求 CSV |
| 精度测试 | GSM8K 格式数据集精度评估 |

## 快速开始

```bash
# 依赖：胶囊/镜像内已带 ais_bench；另需
pip install transformers plotly tabulate requests pandas

git clone https://github.com/gaoyuan777/aisbench-prefix-test-tool.git
cd aisbench-prefix-test-tool
cp config_template_colocated.py config.py   # 混部；PD 分离用 config_template_pd.py
vi config.py                                # 必改 MODEL_NAME / MODEL_PATH / HOST_IP / HOST_PORT
```

`config.py` 不入库（`.gitignore` 已忽略），按模板改以下关键项即可：

```python
MODEL_NAME = "your-model-name"          # 与 vllm serve --served-model-name 一致
MODEL_PATH = "/path/to/model_weights"   # 仅用于读 tokenizer
HOST_IP = "192.168.1.10"                # 混部填服务地址；PD 分离填 proxy 地址
HOST_PORT = "8000"

DATASET_PATH = "/path/to/dataset_dir"   # 数据集生成目录（不存在会自动创建）
WORK_PATH = "/usr/local/python3.12.13/lib/python3.12/site-packages"  # pip show ais-bench-benchmark 的 Location
OUTPUT_DIR = "./outputs"

# 可选：命中率 / 运行时监控 / 五阶段
POD_INFO = []          # ["ip:port", ...]，--prefix_test 命中率查询；留空默认 HOST_IP:HOST_PORT
M_LISTEN_SERVER = ""   # 混部：vLLM 引擎 metrics 端口（不要填 proxy）
# P_LISTEN_SERVER / D_LISTEN_SERVER    # PD 分离形态
# KV_STAGE_LOG_P / KV_STAGE_LOG_D      # PD 五阶段，见下文
```

冒烟验证：

```bash
python3 aisbench_test.py --input_len 64 --output_len 1 --data_num 2 --concurrency 2
```

能产出 `aisbench_result.csv` 即环境就绪。

## 常用用例

```bash
# 1. 普通性能测试
python3 aisbench_test.py --input_len 2048 --output_len 2048 --data_num 160 --concurrency 40 --request_rate 10

# 2. Prefix Cache 命中率（2 种前缀、重复率 50%、dp=2、先 warmup）
python3 aisbench_test.py --input_len 2048 --output_len 2048 --data_num 160 --concurrency 40 \
  --dataset_type prefix_cache --repeat_rate 0.5 --prefix_num 2 --prefix_test --dp 2

# 3. 最小前缀用例（4 条 4k 输入、单并发、全量重复前缀）
python3 aisbench_test.py --input_len 4096 --output_len 1 --data_num 4 --concurrency 1 \
  --test_type stream --repeat_rate 1 --dataset_type prefix_cache --prefix_test --dp 1 \
  --output_dir ./outputs/4k_1_cc_1_data_4_repeat_rate_0

# 4. 不定长 8k~128k（均值 32k）
python3 aisbench_test.py --input_len 32768 --output_len 300 --data_num 32 --concurrency 8 \
  --dataset_type prefix_cache --repeat_rate 90% --length_mean 32768 --length_std 49152 \
  --length_min 8192 --length_max 131072 --prefix_test --dp 2

# 5. 重复 3 轮（CSV 每轮一行，带 round 列）
python3 aisbench_test.py --input_len 2048 --output_len 2048 --data_num 160 --concurrency 40 --repeat 3

# 6. 指定数据集 / 精度测试
python3 aisbench_test.py --dataset "/mnt/data/medium.jsonl" --output_len 20 --concurrency 1024
python3 aisbench_test.py --dataset "/mnt/data/gsm8k.jsonl" --output_len 1024 --concurrency 64 --test_accuracy
```

## 参数

`python3 aisbench_test.py --help` 查看全部。常用：

| 参数 | 说明 |
| --- | --- |
| `--input_len` / `--output_len` | 输入 / 输出 token 数（之和不超过服务 `max_model_len`） |
| `--data_num` / `--concurrency` / `--request_rate` | 数据集条数 / 最大并发 / 请求频率（0 = 不限流） |
| `--test_type` | `stream`（默认）/ `text` |
| `--dataset` | 指定 gsm8k 格式数据集（与自动生成二选一） |
| `--dataset_type` | `normal`（默认）/ `prefix_cache` |
| `--repeat_rate` | 前缀重复率，`"0.5"` 或 `"50%"`（取值 0~1） |
| `--prefix_num` / `--dp` | 前缀种类数 / DP 域数（warmup 条数 = `prefix_num × dp × 2`） |
| `--prefix_test` | 全量前先 warmup（命中率统计仅含全量阶段） |
| `--repeat` | 重复轮数，每轮单独落一行 CSV |
| `--npu_num` | NPU 卡数（折算单卡吞吐） |
| `--output_dir` | 覆盖 config 的 `OUTPUT_DIR` |
| `--timeout` | 单轮 ais_bench 超时秒数（默认不限） |
| `--enable_think` / `--test_accuracy` | 思考模式 / 精度测试 |
| `--length_mean/std/min/max` | 不定长数据集的长度分布 |

## 配置要点

- **两种形态的差异**：混部 `HOST_*` 填服务地址、只配 `M_LISTEN_SERVER`；PD 分离 `HOST_*` 填 **proxy**、配 `P_LISTEN_SERVER`/`D_LISTEN_SERVER`，且 `POD_INFO` 必须填 **P 节点各 DP 域端口**（填错只会统计到 0 命中）
- **`*_LISTEN_SERVER` 填 vLLM 引擎 metrics 端口**，不要填 proxy（通常 404）；多 DP 域逗号分隔；`M_LISTEN_SERVER` 非空时忽略 P/D
- **DP 部署形态决定填几个地址**：单实例内部 DP（`--data-parallel-size N --api-server-count 1`）一个端口返回全部 DP 数据（`engine="0".."N-1"`），填 1 个即可；多 API server / 多实例外部 DP 每端口只含本 DP 域，需列全部端口。判定：`curl -s http://{ip}:{port}/metrics | grep -o 'engine="[0-9]*"' | sort -u`
- **命中率与设定值有偏差**：vLLM 按 block（默认 16 token）整块命中，前缀尾部含唯一 token 的块不可命中，实测命中率略低于 `repeat_rate`；输入越短偏差占比越大

## 产物

| 产物 | 位置 |
| --- | --- |
| 性能结果 | `aisbench_result.csv`（`summarizer` 列标记口径；多轮时 `round` 列标记轮次） |
| 日志 | `aisbench.log`（本次）/ `aisbench_all.log`（累积）/ `outputs/<时间戳>/` |
| 运行时指标 | `outputs/<...>/performances/vllm-api-stream-chat/vllm_pd_runtime_metrics.html`；实时数据 `tail -F pd_metrics_live.txt` |
| 五阶段（可选） | HTML 五阶段图 + `aisbench_result.csv` 5 列均值 + `kv_stage_details.csv` 逐请求明细 |

## PD 五阶段耗时分析（可选）

数据来自 vLLM 的 `REQ_STAGE_METRIC` 日志（P/D 各一条/请求，含 `ts_ms` 毫秒锚点），由上游 draft PR [vllm-project/vllm#62685](https://github.com/vllm-project/vllm/pull/62685) 提供，**需在部署的 vLLM 中携带该补丁**。

```python
KV_STAGE_LOG_P = "/workspace/pd/vllm_p.log"
KV_STAGE_LOG_D = "/workspace/pd/vllm_d.log"
```

两项同时配置才启用；仅 PD 分离形态生效（混部告警忽略）。按日志 offset 增量采集，无需清日志；日志中找不到 `REQ_STAGE_METRIC` 时会明确告警提示补丁缺失。配套的 `KV_XFER_METRIC`（worker 级传输明细）为可选增强，缺失只影响 hover 分解与对应 CSV 列。

## FAQ

1. **"生成数据集失败，请清空 picked ids"**：删除 `picked_ids.txt`（GSM8K 共 1319 题，唯一模式用尽后需重置）。
2. **命中率为空**：仅 `--prefix_test` 统计；检查 `POD_INFO` 是否填引擎端口；`unset http_proxy; curl -s http://{ip}:{port}/metrics | grep prefix` 应返回 `vllm:prefix_cache_*`。
3. **HTML 为空/不刷新**：`*_LISTEN_SERVER` 必须是引擎 metrics 端口；用 `tail -F pd_metrics_live.txt` 确认每秒有数据。
4. **"Can not find a stable stage"**：数据量太小或并发未打满；增大 `--data_num`，或把 `DEFAULT_PERFORMANCE_TEST` 改为 `default_perf`。
5. **五阶段图不生成**：`KV_STAGE_LOG_P/D` 是否都填、日志是否存在；`grep REQ_STAGE_METRIC vllm_p.log` 是否有输出；未配 `*_LISTEN_SERVER` 时不生成 HTML（仍有打屏与 CSV）。
6. **`unrecognized arguments: --num-warmups`**：aisbench 版本过旧，删除 `aisbench_test.py` 中该参数。

## License

见 [LICENSE](LICENSE)。
