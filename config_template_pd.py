# -*- coding: utf-8 -*-
"""PD 分离（Disaggregated）形态 config 模板 —— cp config_template_pd.py config.py 后按环境修改"""

# 数据集存放路径（绝对路径，需提前 mkdir）
DATASET_PATH = "/path/to/dataset_dir"

# ais_bench 工作路径，`pip show ais-bench-benchmark` 查询 Location
WORK_PATH = "/usr/local/python3.11/lib/python3.11/site-packages"

# 服务化模型名（与 vllm serve --served-model-name 一致）
MODEL_NAME = "your-model-name"

# 模型权重路径（读取 tokenizer 用）
MODEL_PATH = "/path/to/model_weights"

# 请求目的地址：PD 分离形态必须填 proxy（负载均衡代理）地址，
# 不要填 P 或 D 节点直连地址，否则请求无法正确分发
HOST_IP = "192.168.1.10"
HOST_PORT = "8000"  # proxy 端口

# 鉴权信息（如服务未开启鉴权则留空）
API_KEY = ""

# 稳态测试改为 "stable_stage"，否则保持 "default_perf"
DEFAULT_PERFORMANCE_TEST = "default_perf"

# aisbench 输出保存路径（可被 --output_dir 覆盖）
OUTPUT_DIR = "./outputs/default"

# --prefix_test 各 DP 域命中率查询地址：
#   prefix cache 命中发生在 P（Prefill）节点，至少列出全部 P 实例的各 DP 域端口
#   注意是 vLLM 引擎端口，不是 proxy 端口
#   格式 ["{ip}:{port}", ...]，留空 [] 则默认 HOST_IP:HOST_PORT（PD 形态下通常不正确，务必填写）
# 示例：P 单机 dp=4，端口 9000-9003
POD_INFO = ["192.168.1.11:9000", "192.168.1.11:9001", "192.168.1.11:9002", "192.168.1.11:9003"]

# ===== 运行时指标监控（可选）=====
# M_LISTEN_SERVER 必须留空，PD 分离监听才会生效（M 优先级最高）
M_LISTEN_SERVER = ""

# P（Prefill）/ D（Decode）节点各 DP 域的 metrics 端口，逗号分隔多个端点
# 填 vLLM 引擎自身的 metrics 端口（不要填 proxy），留空 "" 则不监听该角色
# 示例：P 192.168.1.11 dp=4 端口 9000-9003；D 192.168.1.12 dp=4 端口 10001-10004
P_LISTEN_SERVER = "192.168.1.11:9000,192.168.1.11:9001,192.168.1.11:9002,192.168.1.11:9003"
D_LISTEN_SERVER = "192.168.1.12:10001,192.168.1.12:10002,192.168.1.12:10003,192.168.1.12:10004"
