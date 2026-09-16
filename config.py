# 数据集文件夹路径，需可访问(请使用绝对路径)
DATASET_PATH = "/home/dataset"

# aisbench 工作路径, 为 git clone aisbench 后得到的 benchmark 目录的绝对路径
# 可通过命令 `pip show ais-bench-benchmark` 查询location
WORK_PATH = "/home/benchmark"

# 服务化配置的模型名称
MODEL_NAME = "ds"

# 模型权重路径, 用于读取 tokenizer
MODEL_PATH = "/home/weights/model_weights"

# 请求目的 IP
HOST_IP = "141.xx.xx.xx"

# 请求目的端口
HOST_PORT = "8004"

## 鉴权信息
API_KEY = ""

# 如果使用稳态测试请将该字段设置为 "stable_stage"
DEFAULT_PERFORMANCE_TEST = "default_perf"

# aisbench输出日志保存路径（可被命令行 --output_dir 覆盖）
OUTPUT_DIR = "./outputs/default"

# 各节点信息，格式为 ["{ip}:{port}"]
# 用于 --prefix_test 查询各 dp 域 prefix cache 命中率，不配置默认为 HOST_IP:HOST_PORT
# PD分离场景请填写各个节点的IP和对应dp域的port
# POD_INFO = ["141.xx.xx.11:8000","141.xx.xx.12:8000"]
POD_INFO = []


# 混部运行时指标监听：metrics 地址 IP:端口。填写后测试期间每秒采集 KVCache/队列等指标并生成 HTML，
# 优先走混部采集，忽略下面的 P/D。与 POD_INFO 无关（POD_INFO 仍只用于 --prefix_test）。
M_LISTEN_SERVER = ""   # 例: "172.27.25.14:40099"


# PD分离场景：P（Prefill）监听地址 IP:端口。测试期间每秒采集 KVCache/队列等运行时指标。
# 仅当 M_LISTEN_SERVER 为空时生效。不填写则默认不监听 P。
P_LISTEN_SERVER = ""   # 例: "172.27.26.11:8000"

# PD分离场景：D（Decode）监听地址 IP:端口。测试期间每秒采集 KVCache/队列等运行时指标。
# 仅当 M_LISTEN_SERVER 为空时生效。不填写则默认不监听 D。
D_LISTEN_SERVER = ""   # 例: "172.27.26.12:8001"
