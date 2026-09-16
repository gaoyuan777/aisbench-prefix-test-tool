# aisbench_auto_tools_prefix

基于 [ais_bench](https://github.com/AISBench/benchmark) 的 vLLM 推理服务性能测试工具，支持 **性能测试**、**Prefix Cache 命中率测试** 与 **运行时指标监控可视化**。

> **项目来源**：本项目参考了 [rayn-zzz/aisbench_auto_tools_prefix](https://github.com/rayn-zzz/aisbench_auto_tools_prefix) 项目的部分内容，并在此基础上重新设计开发。

## 功能特性

1. **模型性能测试**：定长/不定长输入、流式/非流式、思考模式、稳态测试
2. **Prefix Cache 性能测试**：自动生成带公共前缀的数据集，先预热后测试，统计各 DP 域命中率
3. **运行时指标监控**：测试期间每秒采集 vLLM `/metrics`，生成单文件交互式 HTML 报告（KV Cache 占用率、各 DP 队列、请求时间轴等），支持混部与 PD 分离两种部署形态
4. **数据集生成**：基于 GSM8K 生成指定长度、指定前缀重复率、指定前缀个数的测试数据集
5. **精度测试**：支持 GSM8K 格式数据集的精度评估

## 环境准备

进入带 [ais_bench](https://github.com/AISBench/benchmark) 的环境（mindie/vllm 镜像均已打包 aisbench），另需安装：

```bash
pip install transformers plotly tabulate requests
```

## 快速开始

```bash
git clone https://github.com/rayn-zzz/aisbench_auto_tools_prefix.git
cd aisbench_auto_tools_prefix

# 1. 修改 config.py（服务地址、模型名、权重路径等）
# 2. 运行测试
python3 aisbench_test.py --input_len 2048 --output_len 2048 --data_num 160 --concurrency 40 --request_rate 10
```

**注：脚本会自动生成数据集，需先在 config.py 的 `DATASET_PATH` 指向的位置创建存放文件夹**，eg：`mkdir /mnt/path_to_store_dataset`

## 配置说明（config.py）

| 字段 | 说明 |
| --- | --- |
| `DATASET_PATH` | 生成数据集的存放文件夹路径（绝对路径） |
| `WORK_PATH` | ais_bench 工作路径，`pip show ais-bench-benchmark` 查询 |
| `MODEL_NAME` | 服务化配置的模型名称 |
| `MODEL_PATH` | 模型权重路径，用于读取 tokenizer |
| `HOST_IP` / `HOST_PORT` | 请求目的地址与端口 |
| `API_KEY` | 鉴权信息（如有） |
| `DEFAULT_PERFORMANCE_TEST` | 稳态测试设置为 `"stable_stage"` |
| `OUTPUT_DIR` | aisbench 输出保存路径（可被 `--output_dir` 覆盖） |
| `POD_INFO` | 各节点 `["{ip}:{port}"]`，用于 `--prefix_test` 查询各 DP 域命中率，不配置默认为 `HOST_IP:HOST_PORT` |
| `M_LISTEN_SERVER` | 混部运行时指标监听地址。填写后每秒采集 + 生成 HTML，优先级最高 |
| `P_LISTEN_SERVER` | PD 分离场景 P（Prefill）监听地址，仅当 `M_LISTEN_SERVER` 为空时生效 |
| `D_LISTEN_SERVER` | PD 分离场景 D（Decode）监听地址，仅当 `M_LISTEN_SERVER` 为空时生效 |

> 三个 `*_LISTEN_SERVER` 均不填写时不启用运行时监控。必须是 vLLM 引擎自己的 metrics 端口，不要填 proxy。

## 命令行参数

`python3 aisbench_test.py --help` 可查看所有参数

| 参数名 | type | 释义 |
| --- | --- | --- |
| `--input_len` | int | 输入长度 |
| `--output_len` | int | 输出长度 |
| `--data_num` | int | 数据集条数 |
| `--concurrency` | int | 系统最大并发数 |
| `--request_rate` | int | 请求频率，默认 0（不限流） |
| `--test_type` | str | text or stream，测试流式 or 非流式，默认 stream |
| `--dataset` | str | 指定测试数据集路径，仅限 gsm8k 格式 |
| `--repeat` | int | 单条命令的测试次数，默认 1。注：数据采集功能只能采集最后一次测试数据 |
| `--enable_think` | bool | DeepSeek V3.1 模型开启 think 功能，默认 false |
| `--test_accuracy` | bool | 测试精度，仅支持 gsm8k 数据集，默认 false |
| `--npu_num` | int | npu 卡数，用于计算单卡吞吐，默认 1 |
| `--dataset_type` | str | normal or prefix_cache，一般数据集 or 带前缀数据集，默认 normal |
| `--prefix_num` | int | 前缀个数，默认 2（warmup 数据量 = dp × prefix_num） |
| `--repeat_rate` | str | 数据集前缀重复率，默认 0.5，支持格式：百分比如 `"50%"` 或小数如 `"0.5"` |
| `--prefix_test` | bool | 是否在全量数据集测试前，先预热前缀（warmup 不限流发送 dp × prefix_num 条），默认 false |
| `--seed` | int | 随机种子，仅适用于生成带前缀数据集，不同 seed 生成的随机 token 不重复，默认 1 |
| `--dp` | int | dp 域数量，默认 1，保证模型推理时前缀会预热到每个 dp 域上 |
| `--output_dir` | str | aisbench 输出目录，覆盖 config.py 中的 `OUTPUT_DIR` |
| `--length_mean` | int | 输入长度均值（不定长数据集） |
| `--length_std` | int | 输入长度标准差（不定长数据集） |
| `--length_min` | int | 输入长度最小值（不定长数据集） |
| `--length_max` | int | 输入长度最大值（不定长数据集） |

## 数据集生成逻辑

- **前缀**：随机挑选一条未使用的 GSM8K 数据，重复/截取到 `input_len × repeat_rate` 长度
- **后缀**：随机挑选一条 GSM8K 数据（可能重复），重复/截取到固定长度
- **完整数据集** = 前缀 + 3 个随机 token（`--seed` 控制） + 后缀

前缀重复率 50% 指每条请求的前 50% token 能在 KV Cache 中命中；前缀个数指前缀的种类数（多种前缀交替出现）。

```text
# 前缀重复率 50%、前缀个数 2 的数据集示例
abc123      ← 前缀 abc
def456      ← 前缀 def
abc789      ← 前缀 abc
def!@#      ← 前缀 def
...
```

## 使用示例

**注：测试 prefix cache 功能需先预热前缀（`--prefix_test`），保证跑全量数据集时存在命中。注意 `input_len + output_len` 不能超过服务的 `max_model_len`。**

1、测试 2k/2k 不带前缀的 gsm8k 数据集性能

```bash
python3 aisbench_test.py --input_len 2048 --output_len 2048 --data_num 160 --concurrency 40 --request_rate 10
```

2、测试 2k/2k 不带前缀的 gsm8k 数据集性能，开启**思考模式**

```bash
python3 aisbench_test.py --input_len 2048 --output_len 2048 --data_num 160 --concurrency 40 --request_rate 10 --enable_think
```

3、测试 2k/2k 带前缀的 gsm8k 数据集性能，前缀个数 2，前缀重复率 50%，dp 2，**先预热前缀**

```bash
python3 aisbench_test.py --input_len 2048 --output_len 2048 --data_num 160 --concurrency 40 --request_rate 10 --dataset_type prefix_cache --repeat_rate 0.5 --prefix_test --dp 2
```

4、测试 2k/2k 带前缀的 gsm8k 数据集性能，前缀个数 3，前缀重复率 73%，不预热前缀直接跑全量数据集

```bash
python3 aisbench_test.py --input_len 2048 --output_len 2048 --data_num 160 --concurrency 40 --request_rate 10 --dataset_type prefix_cache --repeat_rate 73% --seed 200 --prefix_num 3
```

5、测试 8k~128k **不定长**、平均 32k、带前缀的 gsm8k 数据集性能，前缀重复率 90%，dp 2，**先预热前缀**

```bash
python3 aisbench_test.py --input_len 32768 --output_len 300 --data_num 32 --concurrency 8 --request_rate 0 --dataset_type prefix_cache --repeat_rate 90% --prefix_test --dp 2 --length_mean 32768 --length_std 49152 --length_min 8192 --length_max 131072
```

6、测试指定数据集性能（仅限 **gsm8k** 格式）

```bash
python3 aisbench_test.py --dataset "/mnt/path_to_dataset/medium2.jsonl" --output_len 20 --concurrency 1024
```

7、测试指定数据集精度（仅限 **gsm8k** 格式）

```bash
python3 aisbench_test.py --dataset "/mnt/path_to_dataset/precision_dataset.jsonl" --output_len 1024 --concurrency 64 --request_rate 4 --test_accuracy
```

## 运行时指标监控

配置 `M_LISTEN_SERVER`（混部）或 `P_LISTEN_SERVER` + `D_LISTEN_SERVER`（PD 分离）后，测试期间每秒采集 vLLM `/metrics`，测试结束自动生成交互式 HTML 报告：

```text
outputs/<目录>/<时间戳>/performances/vllm-api-stream-chat/vllm_pd_runtime_metrics.html
```

- 实时结果覆盖写入 `pd_metrics_live.txt`（不刷屏），另开终端执行 `tail -F pd_metrics_live.txt` 查看
- 前缀命中率只统计**全量数据集阶段**（warmup 排除），写入 `aisbench_result.csv` 的 Prefix Hit Rate 列及 HTML 标题下方
- `--prefix_test` 的命中率查询仍走 `POD_INFO`，与运行时 HTML 监听无关

### 混部模式（M_LISTEN_SERVER）图表布局

| 图号 | 内容 |
| --- | --- |
| 图1 | 请求时间轴（来自 gsm8k_plot，warmup 虚线 / 全量实线） |
| 图2 | KV Cache 占用率随时间变化（聚合） |
| 图3 | 请求并发数（来自 gsm8k_plot） |
| 图4 | 混部 Running/Waiting/Preempted 队列柱状图（每 1 秒一组） |
| 图5 | 各 DP KV Cache 占用率（每个 DP 一条折线） |
| 图6 | 各 DP Running/Waiting 队列柱状图（每个 DP 一种颜色，实色 Running + 浅色 Waiting 堆叠） |
| 图7 | ais_bench 测试结果表 |

### PD 分离模式（P/D_LISTEN_SERVER）图表布局

| 图号 | 内容 |
| --- | --- |
| 图1 | 请求时间轴（来自 gsm8k_plot） |
| 图2 | KV Cache 占用率随时间变化（聚合） |
| 图3 | 请求并发数（来自 gsm8k_plot） |
| 图4 | P 节点 Running/Waiting/Preempted 队列柱状图（每 1 秒一组） |
| 图5 | P 节点各 DP KV Cache 占用率 |
| 图6 | D 节点 Running/Waiting/Preempted 队列柱状图（每 1 秒一组） |
| 图7 | P 节点各 DP Running/Waiting 队列柱状图 |
| 图8 | D 节点各 DP KV Cache 占用率 |
| 图9 | D 节点各 DP Running/Waiting 队列柱状图 |
| 图10 | ais_bench 测试结果表 |

所有时间轴图表支持联动缩放（拖动任一图横轴，其余图同步），图表底部附参数配置表。

## 结果获取

1. **性能结果**：`aisbench_result.csv`（暂不支持精度结果获取，需在日志中查看）
2. **当前命令 aisbench 日志**：`aisbench.log`
3. **历史命令日志**：`aisbench_all.log` 或 `outputs/<目录>/<时间戳>/aisbench.log`
4. **Prefix Cache 命中率**：见打屏日志及 HTML 标题
5. **运行时指标**：见上文 HTML 路径

## FAQ（常见问题）

### 1、出现 ERROR 日志：生成数据集失败，请清空 picked ids

解决方案：删除 `picked_ids.txt` 文件。

### 2、加载 tokenizer 报错

解决方案：检查当前 transformers 版本是否适配模型，如 GLM5 需更新 mindie/vllm 镜像内 transformers 版本。

### 3、`ais_bench: error: unrecognized arguments: --num-warmups`

解决方案：修改 aisbench_test.py，搜索并删除 `--num-warmups 0`。

※ `--num-warmups` 为 aisbench 最新版本功能，使用时需配置为 0，详情参考 [aisbench github](https://github.com/AISBench/benchmark) 官网。

### 4、常用 shell 固定并发测试脚本

```bash
bs=(1 8 16 24 32 40 48 56)
for i in ${bs[@]}
do
        python3 aisbench_test.py --input_len 8192 --output_len 1 --data_num $(($i * 4)) --concurrency $i --request_rate 0 --dataset_type prefix_cache --repeat_rate 0.5 --seed $i --prefix_num 1 --prefix_test
done
```

### 5、打屏不显示 prefix cache 命中率信息

只有开启 `--prefix_test` 才会打印命中率信息，其余情况可在测试前后分别发送 curl 命令获取 metrics 自行计算：

```bash
curl -s http://{ip_address}:{port}/metrics | grep prefix
```

```text
hit rate = (第二次 prefix_cache_hits - 第一次 prefix_cache_hits) / (第二次 prefix_cache_queries - 第一次 prefix_cache_queries)
```

排查步骤：

1）检查 config.py 里的 `POD_INFO` 是否配置正确。混部配主节点 IP 和 PORT；PD 分离配各节点 IP 和对应 DP 域的 PORT。

2）手动验证是否能正确获取 vllm metrics 信息：

```bash
unset http_proxy
unset https_proxy
curl -s http://{ip_address}:{port}/metrics | grep prefix
```

正常情况返回 `vllm:prefix_cache_queries` 和 `vllm:prefix_cache_hits` 信息。

### 6、运行时指标没有刷新 / HTML 是空的

1）混部：填写 `M_LISTEN_SERVER`。PD 分离：`M_LISTEN_SERVER` 留空，填写 `P_LISTEN_SERVER` / `D_LISTEN_SERVER`（格式 `ip:port`）。混部与 P/D 都填时**只走混部**。`POD_INFO` 只给 `--prefix_test` 用，不要填到运行时监听里。

2）必须是 vLLM 引擎自己的 metrics 端口，不要填 proxy（proxy 的 `/metrics` 常为 404）。

3）实时监控请 `tail -F pd_metrics_live.txt`（该文件每秒覆盖刷新，不是追加）。

4）在测试容器内手动验证：

```bash
curl -s http://{IP}:{port}/metrics | grep -E "kv_cache_usage|gpu_cache_usage|prefix_cache_hits|prefix_cache_queries|num_requests_running|num_requests_waiting|num_preemptions|num_requests_swapped|cache_config_info"
```

## License

见 [LICENSE](LICENSE)。

## 致谢

本项目参考了 [rayn-zzz/aisbench_auto_tools_prefix](https://github.com/rayn-zzz/aisbench_auto_tools_prefix) 项目的部分内容，并在此基础上重新设计开发，感谢原作者的工作。
