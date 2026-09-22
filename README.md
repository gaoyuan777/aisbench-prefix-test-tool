# aisbench-prefix-test-tool

基于 [ais_bench](https://github.com/AISBench/benchmark) 的 vLLM 推理服务测试工具：**性能测试**、**Prefix Cache 命中率测试**、**运行时指标监控可视化**（混部 / PD 分离）。

> 项目来源：参考 [rayn-zzz/aisbench_auto_tools_prefix](https://github.com/rayn-zzz/aisbench_auto_tools_prefix) 部分内容重新设计开发。

## 功能

- **性能测试**：定长/不定长输入、流式/非流式、思考模式、稳态测试
- **Prefix Cache 测试**：自动生成带公共前缀的数据集，warmup 预热后跑全量，统计各 DP 域命中率
- **运行时监控**：测试期间每秒采集 vLLM `/metrics`，生成单文件交互式 HTML（KV 占用、各 DP 队列、请求时间轴，支持联动缩放）
- **数据集生成**：基于 GSM8K，可指定长度、前缀重复率、前缀个数
- **精度测试**：GSM8K 格式数据集精度评估

## 快速开始

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

## config.py 模板（二选一）

| 形态 | 模板文件 | 关键差异 |
| --- | --- | --- |
| 混部 | `config_template_colocated.py` | `HOST_*` 填 vLLM 服务地址；监控只填 `M_LISTEN_SERVER`；`POD_INFO` 填服务地址 |
| PD 分离 | `config_template_pd.py` | `HOST_*` 填 **proxy** 地址；`POD_INFO` 填 **P 节点各 DP 域端口**；监控填 `P_LISTEN_SERVER` / `D_LISTEN_SERVER` |

公共规则：

- `M_LISTEN_SERVER` 优先级最高，非空时忽略 P/D 监听；三个监听全空则不启用运行时监控
- `*_LISTEN_SERVER` 必须是 vLLM 引擎自身的 metrics 端口（多 DP 域逗号分隔），**不要填 proxy**（其 `/metrics` 常为 404）
- `POD_INFO` 只服务 `--prefix_test` 的命中率统计，与运行时监控无关；留空 `[]` 默认 `HOST_IP:HOST_PORT`

## 命令行参数

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

## Prefix Cache 测试逻辑

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

## 使用示例

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

## 运行时指标监控

配置 `*_LISTEN_SERVER` 后，测试结束自动生成：

```text
outputs/<目录>/<时间戳>/performances/vllm-api-stream-chat/vllm_pd_runtime_metrics.html
```

- 实时数据覆盖写入 `pd_metrics_live.txt`，另开终端 `tail -F pd_metrics_live.txt` 查看
- 前缀命中率写入 `aisbench_result.csv` 与 HTML 标题下方

**混部布局（7 图）**：请求时间轴 / KV 占用率（聚合平均）/ 并发数 / 队列柱状图 / 各 DP KV / 各 DP 队列 / 结果表

**PD 分离布局（10 图）**：请求时间轴 / KV 占用率（**P/D 各一条平均线**）/ 并发数 / P 队列 / P 各 DP KV / D 队列 / P 各 DP 队列 / D 各 DP KV / D 各 DP 队列 / 结果表

所有时间轴图表联动缩放，底部附参数配置表。

## 结果获取

| 产物 | 位置 |
| --- | --- |
| 性能结果 | `aisbench_result.csv` |
| 当次日志 | `aisbench.log`（历史：`aisbench_all.log`、`outputs/<目录>/<时间戳>/`） |
| 命中率 | 打屏日志 + `aisbench_result.csv` + HTML 标题 |
| 运行时指标 | 上文 HTML 路径 |

## FAQ

**1. 报错"生成数据集失败，请清空 picked ids"**：删除 `picked_ids.txt`。

**2. 命中率不显示/为空**：仅 `--prefix_test` 时统计命中率；检查 `POD_INFO` 配置（混部填服务地址，PD 填 P 节点各 DP 域端口）；手动验证 `unset http_proxy; curl -s http://{ip}:{port}/metrics | grep prefix` 能否返回 `vllm:prefix_cache_*`。

**3. 运行时 HTML 为空/不刷新**：`*_LISTEN_SERVER` 必须是 vLLM 引擎 metrics 端口而非 proxy；混部与 P/D 都填时只走混部；用 `tail -F pd_metrics_live.txt` 看实时数据（每秒覆盖刷新）。

**4. `unrecognized arguments: --num-warmups`**：aisbench 版本过旧，删除 `aisbench_test.py` 中的 `--num-warmups 0`。

**5. tokenizer 加载失败**：检查 transformers 版本与模型适配性，部分模型需 `trust_remote_code=True`。

## License

见 [LICENSE](LICENSE)。

## 致谢

本项目参考了 [rayn-zzz/aisbench_auto_tools_prefix](https://github.com/rayn-zzz/aisbench_auto_tools_prefix) 项目的部分内容，并在此基础上重新设计开发，感谢原作者的工作。
