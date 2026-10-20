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

# 性能统计口径：stable_stage（稳态窗口，默认）/ default_perf（全量请求）
# warmup 阶段始终固定 default_perf（数据量太小无法形成稳态窗口），此处只控制全量阶段
DEFAULT_PERFORMANCE_TEST = "stable_stage"

# aisbench 输出保存路径（可被 --output_dir 覆盖）
OUTPUT_DIR = "./outputs/default"

# PD 请求五阶段耗时分析（可选）：
#   vllm 侧需已合入 REQ_STAGE_METRIC / KV_XFER_METRIC 上报（见配套开发指南）
#   P/D 节点 vllm 日志文件路径；留空 "" 则不启用（不生成图 11，其余功能不受影响）
#   单机部署（P/D 同容器）两路径均填本机路径；跨机部署 D 侧日志需挂载/拷贝到本机可访问路径
KV_STAGE_LOG_P = ""   # 例: "/workspace/pd/vllm_p.log"
KV_STAGE_LOG_D = ""   # 例: "/workspace/pd/vllm_d.log"

# --prefix_test 各 DP 域命中率查询地址：
#   prefix cache 命中发生在 P（Prefill）节点，注意是 vLLM 引擎端口，不是 proxy 端口
#   格式 ["{ip}:{port}", ...]，留空 [] 则默认 HOST_IP:HOST_PORT（PD 形态下通常不正确，务必填写）
#   填法同样受 DP 部署形态影响（形态说明见下方 P_LISTEN_SERVER）：
#     形态 A（单实例内部 DP）填 1 个地址即可；形态 B/C 列出全部端口
# 示例：P 单机 dp=4 多实例，端口 9000-9003
POD_INFO = ["192.168.1.11:9000", "192.168.1.11:9001", "192.168.1.11:9002", "192.168.1.11:9003"]

# ===== 运行时指标监控（可选）=====
# M_LISTEN_SERVER 必须留空，PD 分离监听才会生效（M 优先级最高）
M_LISTEN_SERVER = ""

# P（Prefill）/ D（Decode）节点的 metrics 端点，留空 "" 则不监听该角色
# 填 vLLM 引擎自身的 metrics 端口（不要填 proxy），多端点逗号分隔
#
# 按 DP 部署形态填写（三种形态详见 README「DP 部署形态与 metrics 配置」）：
#   A. 单实例内部 DP：vllm serve --data-parallel-size 4 --api-server-count 1
#      一个 metrics 端口返回全部 DP 域数据（prometheus engine 标签区分）→ 只填 1 个地址
#   B. 多 API server：--data-parallel-size 4 --api-server-count 4（端口递增）
#      每端口仅本 DP 域 → 列出全部端口
#   C. 多实例外部 DP：launch_online_dp.py（每 DP 一个独立 vllm serve 进程，端口递增）
#      每端口仅本实例 → 列出全部端口
# 判定命令（服务运行时执行）：
#   curl -s http://{ip}:{port}/metrics | grep -o 'engine="[0-9]*"' | sort -u
#   返回 dp 数行（engine="0"~"3"）→ 形态 A 填 1 个地址；返回 1 行/空 → 形态 B/C 列全部端口
P_LISTEN_SERVER = "192.168.1.11:9000,192.168.1.11:9001,192.168.1.11:9002,192.168.1.11:9003"
D_LISTEN_SERVER = "192.168.1.12:10001,192.168.1.12:10002,192.168.1.12:10003,192.168.1.12:10004"
