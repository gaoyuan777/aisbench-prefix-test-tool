# -*- coding: utf-8 -*-
"""混部（Hybrid）形态 config 模板 —— cp config_template_colocated.py config.py 后按环境修改"""

# 数据集存放路径（绝对路径，需提前 mkdir）
DATASET_PATH = "/path/to/dataset_dir"

# ais_bench 工作路径，`pip show ais-bench-benchmark` 查询 Location
WORK_PATH = "/usr/local/python3.11/lib/python3.11/site-packages"

# 服务化模型名（与 vllm serve --served-model-name 一致）
MODEL_NAME = "your-model-name"

# 模型权重路径（读取 tokenizer 用）
MODEL_PATH = "/path/to/model_weights"

# 请求目的地址：混部形态直填 vLLM 服务地址（单实例多 DP 时填对外暴露的统一端口）
HOST_IP = "192.168.1.10"
HOST_PORT = "8000"

# 鉴权信息（如服务未开启鉴权则留空）
API_KEY = ""

# 稳态测试改为 "stable_stage"，否则保持 "default_perf"
DEFAULT_PERFORMANCE_TEST = "default_perf"

# aisbench 输出保存路径（可被 --output_dir 覆盖）
OUTPUT_DIR = "./outputs/default"

# --prefix_test 各 DP 域命中率查询地址：
#   单实例混部填服务地址本身即可；多实例/多 DP 域端口时全部列出
#   格式 ["{ip}:{port}", ...]，留空 [] 则默认 HOST_IP:HOST_PORT
POD_INFO = ["192.168.1.10:8000"]

# ===== 运行时指标监控（可选）=====
# 填 vLLM 引擎自身的 metrics 端口（不要填 proxy），测试期间每秒采集并生成 HTML 报告
# 多 DP 域端口用逗号分隔；留空 "" 则不启用监控
M_LISTEN_SERVER = "192.168.1.10:8000"

# 混部形态下以下两项留空（M_LISTEN_SERVER 优先级最高，填了 P/D 也不会生效）
P_LISTEN_SERVER = ""
D_LISTEN_SERVER = ""
