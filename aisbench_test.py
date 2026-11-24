import os, errno
import sys
import argparse
import re
import logging
import shlex
import signal
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import *
from generate_dataset import *
from save_file import get_data, save_csv, save_log, merge_extra_metrics_into_csv
from cal_prefix_hit_rate import *
from kv_stage_analyzer import KVStageAnalyzer
from vllm_pd_monitor import (
    PDMetricsMonitor,
    is_full_dataset_phase,
    resolve_plot_output_dir,
    role_label,
    find_aisbench_plot_html,
    plot_label_from_phase,
)

# warmup 阶段固定使用的统计口径：数据量仅 prefix_num*dp*2 条且 output_len=1，
# 无法形成稳态窗口（stable_stage 会报 Can not find a stable stage 并中止），必须用全量口径
WARMUP_SUMMARIZER = "default_perf"

# main 块注入变量的模块级默认值（保证函数可独立调用/单测）
test_accuracy = False
enable_think = False
api_test_type = "VLLMCustomAPIChatStream"
api_test_abbr = "vllm-api-stream-chat"
request_rate = "0"
dp = 1

# config 模板占位符：仍为这些字符串说明用户未修改 config.py
_TEMPLATE_PLACEHOLDERS = ["/path/to", "your-model-name", "141.xx.xx.xx", "192.168.1.10"]


def _check_config_placeholder(name, value):
    """启动前快速校验 config 项是否仍为模板占位符，命中则致命退出（不发网络请求）。"""
    for placeholder in _TEMPLATE_PLACEHOLDERS:
        if placeholder in str(value):
            logging.error(f"配置项 {name} 仍是模板占位符（含 '{placeholder}'）：{value}，请修改 config.py")
            sys.exit(1)


def _validate_cli_args(parser, args):
    """解析后立即校验数值与参数组合，非法时以清晰的错误信息退出。"""
    if args.input_len < 1:
        parser.error("--input_len 必须 >= 1")
    if args.output_len < 1:
        parser.error("--output_len 必须 >= 1")
    if args.data_num < 1:
        parser.error("--data_num 必须 >= 1")
    if args.concurrency < 1:
        parser.error("--concurrency 必须 >= 1")
    if args.npu_num < 1:
        parser.error("--npu_num 必须 >= 1")
    if args.dp < 1:
        parser.error("--dp 必须 >= 1")
    if args.prefix_num < 1:
        parser.error("--prefix_num 必须 >= 1")
    if args.repeat < 1:
        parser.error("--repeat 必须 >= 1")
    if args.seed < 0:
        parser.error("--seed 必须 >= 0")

    # request_rate 保持 str（兼容 "0"/"10" 等），但必须是数字且 >= 0
    try:
        request_rate_value = float(args.request_rate)
    except (TypeError, ValueError):
        parser.error(f"--request_rate 必须是数字，收到: {args.request_rate}")
    if request_rate_value < 0:
        parser.error("--request_rate 必须 >= 0")

    # repeat_rate 支持 "50%" 或 "0.5"，必须是 [0,1] 内数字
    try:
        repeat_rate_value = parse_prefix_ratio(args.repeat_rate)
    except (TypeError, ValueError) as exc:
        parser.error(f"--repeat_rate 非法: {exc}")

    # prefix_test 仅在 prefix_cache 形态下生效
    if args.prefix_test and args.dataset_type != "prefix_cache":
        logging.warning("--prefix_test 仅在 --dataset_type prefix_cache 时生效，当前已忽略")

    # --repeat 的多轮循环只存在于普通数据集分支，prefix_cache 会忽略该参数
    if args.dataset_type == "prefix_cache" and args.repeat > 1:
        logging.warning("--repeat 仅在 --dataset_type normal 时生效，prefix_cache 模式将忽略该参数（只跑一轮）")

    # accuracy 模式不写 perf 日志，多轮循环拿不到指标，直接拒绝以免产出误导数据
    if args.test_accuracy and args.repeat > 1:
        parser.error("--test_accuracy 与 --repeat>1 暂不兼容：精度测试无性能日志，无法按轮落盘；请去掉 --repeat")

    # prefix_cache + prefix_test 需要工具生成的前缀数据集，不能同时指定外部 --dataset
    if args.dataset_type == "prefix_cache" and args.prefix_test and args.dataset != "none":
        parser.error(
            "--dataset_type prefix_cache 与 --prefix_test 不能同时指定 --dataset："
            "预热需要由工具生成的公共前缀数据集，请省略 --dataset 或去掉 --prefix_test"
        )

    # 前缀预算：repeat_rate<1 时前缀后还需 3 个唯一 token，剩余长度必须为正
    if args.dataset_type == "prefix_cache" and repeat_rate_value < 1:
        prefix_len = int(args.input_len * repeat_rate_value)
        if args.input_len <= prefix_len + 3:
            parser.error(
                f"--input_len={args.input_len} 相对 --repeat_rate={args.repeat_rate} 过小："
                f"前缀占用 {prefix_len} 个 token，还要 3 个唯一 token，剩余长度需 >= 1；"
                f"请增大 --input_len 或减小 --repeat_rate"
            )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_len', type=int, default=3500, help="input token length")
    parser.add_argument("--output_len", type=int, default=1500, help="output token length")
    parser.add_argument("--data_num", type=int, default=8192, help="dataset number")
    parser.add_argument("--concurrency", type=int, default=2048, help="max concurrency")
    parser.add_argument("--request_rate", type=str, default="0", help="request rate")
    parser.add_argument("--test_type", type=str, default="stream", choices=["text", "stream"], help="text or stream")
    parser.add_argument("--dataset", type=str, default="none", help="dataset path")
    parser.add_argument("--repeat", type=int, default=1, help="number of test repeat times")
    parser.add_argument("--enable_think", action='store_true', default=False, help="enable thinking for ds v3.1")
    parser.add_argument("--test_accuracy", action='store_true', default=False, help="test accuracy")
    parser.add_argument("--npu_num", type=int, default=1, help="npu numbers")
    parser.add_argument("--dataset_type", type=str, default="normal", choices=["normal", "prefix_cache"], help="normal or prefix_cache")
    parser.add_argument("--prefix_num", type=int, default=1, help="prefix numbers")
    parser.add_argument("--repeat_rate", type=str, default="0", help="dataset repeat rate")
    parser.add_argument("--prefix_test", action='store_true', default=False, help="test prefix dataset firstly")
    parser.add_argument("--seed", type=int, default=1, help="dataset random seed")
    parser.add_argument("--dp", type=int, default=1, help="dp size")
    parser.add_argument("--length_mean", type=int, default=None, help="gaussian mean for variable length")
    parser.add_argument("--length_std", type=float, default=None, help="gaussian std for variable length")
    parser.add_argument("--length_min", type=int, default=None, help="min length for uniform range or gaussian clip")
    parser.add_argument("--length_max", type=int, default=None, help="max length for uniform range or gaussian clip")
    parser.add_argument("--output_dir", type=str, default=None, help="aisbench output directory, overrides config OUTPUT_DIR if provided")
    parser.add_argument("--timeout", type=float, default=None, help="单阶段 ais_bench 子进程超时秒数，默认 None 表示不限制")
    args = parser.parse_args()
    _validate_cli_args(parser, args)
    return args

def symlink_force(target, link_name):
    logging.info(f"make symlink: {link_name} ==> {target}")
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def create_gsm8k_dataset(dataset_type, input_len, data_num, model_path, dataset_path, prefix_num, repeat_rate, seed,
                         length_mean=None, length_std=None, length_min=None, length_max=None):
    if not os.path.exists(dataset_path):
        logging.warning(f"dataset work path {dataset_path} not exist. creating it automatically.")
        os.makedirs(dataset_path, exist_ok=True)
        logging.info(f"dataset work path {dataset_path} created.")

    if dataset_type == "prefix_cache":
        prefix_jsonl_path, dataset_jsonl_path = create_multi_prefix_dataset(model_path,input_len,data_num,dataset_path,1,dp,repeat_rate,seed,prefix_num,
                                                                             length_mean, length_std, length_min, length_max)
        logging.info("[完成] 数据集已生成：")
        logging.info(f"  - 公共前缀：{prefix_jsonl_path}  (行数={prefix_num*dp*2}, 前缀数={prefix_num})")
        logging.info(f"  - 数据集：  {dataset_jsonl_path} (行数={data_num})")
        logging.info("[信息] 配置：")
        logging.info(f"  tokens(单条长度)={input_len}, prefix_ratio(前缀重复率)={repeat_rate}")
        if length_mean is not None and length_std is not None:
            logging.info(f"  length_dist=gaussian(mean={length_mean}, std={length_std}, min={length_min}, max={length_max})")
        elif length_min is not None and length_max is not None:
            logging.info(f"  length_dist=uniform_int([{length_min}, {length_max}])")
        else:
            logging.info("  length_dist=fixed")
    else:
        # 使用与 create_multi_prefix_dataset 完全一致的命名逻辑（含变长数据集），
        # 避免定长硬编码名字导致变长数据集永远找不到、并误用同名的陈旧文件
        dataset_jsonl_path = expected_dataset_path(
            model_path, input_len, data_num, dataset_path, 0, 0,
            length_mean, length_std, length_min, length_max)
        dataset_name = os.path.basename(dataset_jsonl_path)
        logging.info(f"dataset_name: {dataset_name}")
        prefix_jsonl_path = ""
        # 判断数据集是否存在
        if not os.path.exists(dataset_jsonl_path):
            logging.warning(f"Dataset {dataset_name} is not exist. Start create dataset")
            # create_data(input_len, data_num, model_path, dataset_path)
            prefix_jsonl_path, dataset_jsonl_path = create_multi_prefix_dataset(model_path,input_len,data_num,dataset_path,0,dp,0,seed,prefix_num,
                                                                                 length_mean, length_std, length_min, length_max)
            logging.info(f"Dataset {dataset_name} created.")
        else:
            logging.info(f"Dataset {dataset_name} exist.")
    return prefix_jsonl_path, dataset_jsonl_path

def generate_aisbench_command(summarizer):
    # 对 shell 插值参数做转义，避免路径含空格或注入
    work_dir = shlex.quote(str(OUTPUT_DIR))
    summarizer_arg = shlex.quote(str(summarizer))
    if test_accuracy:
        ais_bench_cmd = f"ais_bench --models vllm_api_chat_temp --datasets gsm8k_gen_0_shot_cot_str_perf --work-dir {work_dir} --dump-eval-details"
    else:
        ais_bench_cmd = f"ais_bench --models vllm_api_chat_temp --datasets gsm8k_gen_0_shot_cot_str_perf --mode perf --summarizer {summarizer_arg} --work-dir {work_dir} --debug --num-warmups 0 2>&1 | tee -a aisbench.log | tee -a aisbench_all.log"
    return ais_bench_cmd

def generate_test_dataset(src_file, dst_dir):
    dst_file = os.path.join(dst_dir, "test.jsonl")
    logging.info(f"src_file: {src_file}")
    logging.info(f"dst_file: {dst_file}")
    # 使用软连接
    symlink_force(src_file, dst_file)
    return

def save_result(request_rate, npu_num, extra_metrics=None, summarizer="default_perf", round_index=None):
    aisbench_log_dir = "aisbench.log"
    filename = "aisbench_result.csv"
    ans, log_dir=get_data(aisbench_log_dir,request_rate,npu_num)
    if extra_metrics:
        ans.update(extra_metrics)
    ans["summarizer"] = summarizer
    if round_index is not None:
        ans["round"] = round_index
    save_log(aisbench_log_dir, log_dir)
    if not save_csv(ans, filename):
        logging.error(f"结果写入 {filename} 失败（详见上方错误），本次数据未落盘")

def modify_aisbench_api(concurrency, output_len, request_rate_param=None):
    if request_rate_param is None:
        request_rate_param = request_rate
    if test_accuracy:
        generation_kwargs = "temperature=0.6,\n\t\t\ttop_p = 0.95"
    else:
        generation_kwargs = "temperature=0,\n\t\t\tignore_eos=True"
    if enable_think:
        generation_kwargs = generation_kwargs + ",\n\t\t\tchat_template_kwargs={\"enable_thinking\": True}"
    generation_kwargs = generation_kwargs.expandtabs(4)
    # 按原顺序做字面替换；用 lambda 保证含反斜杠/正则转义的值原样写入
    replacements = [
        ("model_path_for_replace", MODEL_PATH),
        ("model_name_for_replace", MODEL_NAME),
        ("api_key_for_replace", API_KEY),
        ("rr_for_replace", request_rate_param),
        ("test_type_for_replace", api_test_type),
        ("test_abbr_for_replace", api_test_abbr),
        ("ip_for_replace", HOST_IP),
        ("port_for_replace", HOST_PORT),
        ("outputlen_for_replace", output_len),
        ("concurrency_for_replace", concurrency),
        ("generation_kwargs_for_replace", generation_kwargs),
    ]
    logging.info("Api config file:")
    with open("default_api.py", 'r') as file_default, open("temp_api.py", 'w') as file_temp:
        for ss in file_default:
            tt = ss
            for token, value in replacements:
                tt = re.sub(token, lambda m, v=str(value): v, tt)
            logging.debug(tt)
            file_temp.write(tt)
    symlink_force(
        os.path.join(os.getcwd(), "temp_api.py"),
        os.path.join(WORK_PATH, "ais_bench/benchmark/configs/models/vllm_api/vllm_api_chat_temp.py")
    )

def parse_endpoint(pod):
    """解析 ip:port（含 [ipv6]:port 与无括号 ipv6:port），非法格式抛 ValueError。"""
    text = str(pod).strip()
    if text.startswith('['):
        end = text.find(']')
        if end == -1 or not text[end + 1:].startswith(':'):
            raise ValueError(f"POD_INFO 条目格式应为 ip:port，收到: {pod}")
        ip = text[1:end]
        port = text[end + 2:]
    else:
        if ':' not in text:
            raise ValueError(f"POD_INFO 条目格式应为 ip:port，收到: {pod}")
        ip, port = text.rsplit(':', 1)
    if not ip or not port:
        raise ValueError(f"POD_INFO 条目格式应为 ip:port，收到: {pod}")
    return ip, port

def _fetch_pod_metrics(pod):
    ip, port = parse_endpoint(pod)
    query_token, query_external = get_prefix_queries_total(ip, port)
    hit_token, hit_external = get_prefix_hits_total(ip, port)
    return pod, query_token, query_external, hit_token, hit_external

def get_pod_metrics_info(pod_info):
    query_tokens, query_tokens_external, hit_tokens, hit_tokens_external = {}, {}, {}, {}
    with ThreadPoolExecutor(max_workers=len(pod_info)) as executor:
        futures = [executor.submit(_fetch_pod_metrics, pod) for pod in pod_info]
        for future in as_completed(futures):
            try:
                pod, q_n, q_e, h_n, h_e = future.result()
            except Exception as error:
                logging.warning(f"获取 POD 指标失败，跳过该 POD: {error}")
                continue
            query_tokens[pod] = q_n
            query_tokens_external[pod] = q_e
            hit_tokens[pod] = h_n
            hit_tokens_external[pod] = h_e
    return query_tokens, query_tokens_external, hit_tokens, hit_tokens_external

def cal_prefix_hit_info(query_tokens, query_tokens_external, hit_tokens, hit_tokens_external, query_tokens_new,
                        query_tokens_external_new, hit_tokens_new, hit_tokens_external_new):
    if not query_tokens:
        logging.warning("No prefix cache metrics found.")
        return

    # 输出格式
    line_width = 118
    scope_width, hbm_rate_width, hbm_detail_width, external_rate_width = 35, 19, 21, 21
    headers = ["scope", "hbm_hit_rate", "hbm(hit/query)", "external_hit_rate", "external(hit/query)"]
    header = (f"{headers[0]:<{scope_width}}{headers[1]:<{hbm_rate_width}}"
              f"{headers[2]:<{hbm_detail_width}}{headers[3]:<{external_rate_width}}{headers[4]}")

    def format_rate(hits, queries):
        # 计数器重置/引擎重启会出现负增量，queries==0 时命中率无定义，统一显示 N/A
        if queries <= 0 or hits < 0 or hits > queries:
            return "N/A"
        return format(hits / queries, ".2%")

    neg_warned_scopes = set()

    def build_row(scope, hbm_hits, hbm_queries, external_hits, external_queries):
        if (hbm_hits < 0 or external_hits < 0) and scope not in neg_warned_scopes:
            neg_warned_scopes.add(scope)
            logging.warning(
                f"检测到负的命中增量（计数器重置或引擎重启？） scope={scope}: "
                f"hbm hits delta={hbm_hits}, external hits delta={external_hits}"
            )
        return (f"{scope:<{scope_width}}{format_rate(hbm_hits, hbm_queries):<{hbm_rate_width}}"
                f"{f'{hbm_hits}/{hbm_queries}':<{hbm_detail_width}}"
                f"{format_rate(external_hits, external_queries):<{external_rate_width}}"
                f"{external_hits}/{external_queries}")

    total_hbm_queries = total_hbm_hits = total_external_queries = total_external_hits = 0
    pod_sections, skipped_pods = [], []

    for pod, engines in sorted(query_tokens.items()):
        # 八类指标必须全部存在，且 engine_id 完全一致；
        # 否则无法可靠计算"测试后累计值 - 测试前累计值"，跳过该 POD
        metrics = [query_tokens.get(pod), query_tokens_external.get(pod), hit_tokens.get(pod),
                   hit_tokens_external.get(pod), query_tokens_new.get(pod), query_tokens_external_new.get(pod),
                   hit_tokens_new.get(pod), hit_tokens_external_new.get(pod)]

        if not all(metrics):
            skipped_pods.append(pod)
            continue

        engine_ids = set(engines)
        if any(set(metric) != engine_ids for metric in metrics[1:]):
            skipped_pods.append(pod)
            continue

        # 按 engine_id 升序遍历，为每个 engine 单独生成一行
        engine_rows = []
        for engine_id in sorted(engine_ids):
            engine_scope = f"{pod}/engine{engine_id}"
            engine_hbm_queries = query_tokens_new[pod][engine_id] - query_tokens[pod][engine_id]
            engine_hbm_hits = hit_tokens_new[pod][engine_id] - hit_tokens[pod][engine_id]
            engine_external_queries = query_tokens_external_new[pod][engine_id] \
                - query_tokens_external[pod][engine_id]
            engine_external_hits = hit_tokens_external_new[pod][engine_id] \
                - hit_tokens_external[pod][engine_id]
            engine_rows.append(build_row(engine_scope, engine_hbm_hits, engine_hbm_queries,
                                         engine_external_hits, engine_external_queries))

        # 合并当前 POD 下所有 engine_id 的指标增量，用于 ALL_PODS 汇总
        pod_hbm_queries = sum(query_tokens_new[pod][engine_id] - query_tokens[pod][engine_id]
                              for engine_id in engine_ids)
        pod_hbm_hits = sum(hit_tokens_new[pod][engine_id] - hit_tokens[pod][engine_id]
                           for engine_id in engine_ids)
        pod_external_queries = sum(query_tokens_external_new[pod][engine_id]
                                   - query_tokens_external[pod][engine_id] for engine_id in engine_ids)
        pod_external_hits = sum(hit_tokens_external_new[pod][engine_id]
                                - hit_tokens_external[pod][engine_id] for engine_id in engine_ids)

        # 每个 POD 按 engine 升序输出，POD 之间用分隔线隔开
        pod_sections.append(engine_rows)
        total_hbm_queries += pod_hbm_queries
        total_hbm_hits += pod_hbm_hits
        total_external_queries += pod_external_queries
        total_external_hits += pod_external_hits

    if not pod_sections:
        logging.warning(f"No valid prefix cache metrics found. Skipped PODs: {', '.join(skipped_pods)}")
        return

    all_pods_row = build_row("ALL_PODS", total_hbm_hits, total_hbm_queries,
                             total_external_hits, total_external_queries)
    skipped_pods_line = f"Skipped PODs: {', '.join(skipped_pods)}" if skipped_pods else None

    # 控制台只输出所有有效 POD 的汇总
    console_lines = ["=" * line_width, "Prefix cache hit summary", "=" * line_width, header,
                     "-" * line_width, all_pods_row, "=" * line_width]

    # 日志输出每个 POD 下所有 engine 的详细信息，POD 之间用分隔线隔开
    log_lines = ["=" * line_width, "Prefix cache hit detail", "=" * line_width, header,
                 "-" * line_width]
    for idx, section in enumerate(pod_sections):
        if idx > 0:
            log_lines.append("-" * line_width)
        log_lines.extend(section)
    log_lines.extend(["-" * line_width, all_pods_row, "=" * line_width])

    if skipped_pods_line:
        console_lines.append(skipped_pods_line)
        log_lines.append(skipped_pods_line)

    print("\n".join(console_lines))

    try:
        with open("aisbench.log", "a", encoding="utf-8") as log_file:
            log_file.write("\n" + "\n".join(log_lines) + "\n")
        with open("aisbench_all.log", "a", encoding="utf-8") as all_log_file:
            all_log_file.write("\n" + "\n".join(log_lines) + "\n")
    except OSError as error:
        logging.error(f"Failed to write prefix cache information to log files: {error}")

def setup_logging():
    """配置 Python logging：同时输出到控制台、aisbench.log 和 aisbench_all.log"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 清除已有 handlers（避免重复），但保留会输出到控制台的 StreamHandler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 写入 aisbench.log（本次运行日志）
    file_handler = logging.FileHandler("aisbench.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 写入 aisbench_all.log（所有运行累积日志）
    all_file_handler = logging.FileHandler("aisbench_all.log", mode="a", encoding="utf-8")
    all_file_handler.setFormatter(formatter)
    logger.addHandler(all_file_handler)

if __name__ == '__main__':
    args = parse_arguments()
    input_len = args.input_len
    output_len = args.output_len
    data_num = args.data_num
    concurrency = args.concurrency
    request_rate = args.request_rate
    test_type = args.test_type
    dataset_path_input = args.dataset
    test_times = args.repeat
    enable_think = args.enable_think
    test_accuracy = args.test_accuracy
    npu_num = args.npu_num
    prefix_num = args.prefix_num
    repeat_rate = parse_prefix_ratio(args.repeat_rate)
    prefix_test = args.prefix_test
    dataset_type = args.dataset_type
    seed = args.seed
    dp = args.dp
    length_mean = args.length_mean
    length_std = args.length_std
    length_min = args.length_min
    length_max = args.length_max
    timeout = args.timeout

    # 命令行 --output_dir 覆盖 config.py 中的 OUTPUT_DIR（规范化为绝对路径）
    if args.output_dir:
        OUTPUT_DIR = os.path.abspath(args.output_dir)
        logging.info(f"OUTPUT_DIR overridden by CLI: {OUTPUT_DIR}")

    # 初始化日志文件：清空 aisbench.log（新运行覆盖），aisbench_all.log 添加分隔符（续写）
    with open("aisbench.log", "w", encoding="utf-8") as f:
        f.write("")
    with open("aisbench_all.log", "a", encoding="utf-8") as f:
        f.write(f"\n\n{'='*50}\n")
        f.write(f"New run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n\n")

    # 配置日志输出到控制台 + aisbench.log + aisbench_all.log
    setup_logging()
    logging.info("aisbench.log has been cleared for this run.")
    logging.info("aisbench_all.log separator added for this run.")

    # 启动前快速校验 config：模板占位符 / 路径存在性（无网络依赖）
    _check_config_placeholder("MODEL_PATH", MODEL_PATH)
    if dataset_path_input == "none":
        # 仅在工具自动生成数据集时才用到 DATASET_PATH；外部 --dataset 场景不应因其未改而退出
        _check_config_placeholder("DATASET_PATH", DATASET_PATH)
    _check_config_placeholder("WORK_PATH", WORK_PATH)
    if not os.path.exists(MODEL_PATH):
        logging.error(f"MODEL_PATH 不存在：{MODEL_PATH}，请修改 config.py 指向真实的模型权重/tokenizer 路径")
        sys.exit(1)
    if not os.path.isdir(WORK_PATH):
        derived_work_path = None
        try:
            import ais_bench
            derived_work_path = os.path.dirname(ais_bench.__file__)
        except Exception as error:
            logging.warning(f"尝试通过 import ais_bench 推导 WORK_PATH 失败: {error}")
        if derived_work_path and os.path.isdir(derived_work_path):
            WORK_PATH = derived_work_path
            logging.warning(f"WORK_PATH 不存在，已根据 ais_bench 安装位置自动修正为：{WORK_PATH}")
        else:
            logging.error(
                f"WORK_PATH 不存在：{WORK_PATH}，且无法自动推导。"
                f"请执行 'pip show ais-bench-benchmark' 查看安装路径并修改 config.py 的 WORK_PATH"
            )
            sys.exit(1)

    # 变长参数校验
    if (length_mean is None) ^ (length_std is None):
        raise ValueError("length_mean 和 length_std 必须同时提供或同时不提供")
    if (length_min is None) ^ (length_max is None):
        raise ValueError("length_min 和 length_max 必须同时提供或同时不提供")
    if length_mean is not None and length_mean < 1:
        raise ValueError("length_mean 必须 >= 1")
    if length_std is not None and length_std < 0:
        raise ValueError("length_std 必须 >= 0")
    if length_min is not None and length_min < 1:
        raise ValueError("length_min 必须 >= 1")
    if length_max is not None and length_max < 1:
        raise ValueError("length_max 必须 >= 1")

    logging.info(f"input token length: {input_len}")
    logging.info(f"output token length: {output_len}")
    logging.info(f"number of dataset: {data_num}")
    logging.info(f"concurrency: {concurrency}")
    logging.info(f"request rate: {request_rate}")
    logging.info(f"test type: {test_type}")
    logging.info(f"test_times: {test_times}")
    logging.info(f"v3.1 enable_think: {enable_think}")
    logging.info(f"accuracy test: {test_accuracy}")
    logging.info(f"npu numbers: {npu_num}")
    logging.info(f"prefix numbers: {prefix_num}")
    logging.info(f"dataset repeat rate: {repeat_rate}")
    logging.info(f"test prefix dataset: {prefix_test}")
    logging.info(f"dataset type: {dataset_type}")
    logging.info(f"seed: {seed}")
    logging.info(f"dp size: {dp}")
    logging.info(f"length_mean: {length_mean}")
    logging.info(f"length_std: {length_std}")
    logging.info(f"length_min: {length_min}")
    logging.info(f"length_max: {length_max}")

    # 区分流式和非流式（--test_type 已由 argparse choices 限定为 text/stream）
    if test_type == "text":
        api_test_type = "VLLMCustomAPIChat"
        api_test_abbr = "vllm-api-general-chat"
    else:
        api_test_type = "VLLMCustomAPIChatStream"
        api_test_abbr = "vllm-api-stream-chat"

    if dataset_path_input == "none":
        src_file_prefix,src_file_data = create_gsm8k_dataset(dataset_type, input_len, data_num, MODEL_PATH, DATASET_PATH, prefix_num, repeat_rate, seed,
                                                              length_mean, length_std, length_min, length_max)
    else:
        # 指定数据集路径逻辑
        if not os.path.exists(dataset_path_input):
            logging.error(f"Dataset {dataset_path_input} is not exist.")
            sys.exit(1)
        src_file_data = dataset_path_input
        src_file_prefix = ""

    dst_dir = os.path.join(WORK_PATH, "ais_bench/datasets/gsm8k")

    # 判断 aisbench 的 gsm8k 文件夹是否存在在
    if not os.path.exists(dst_dir):
        logging.info("dataset work path not exist. creating.")
        os.makedirs(dst_dir)
        logging.info("dataset work path created.")
    # 判断 aisbench 的 gsm8k 文件夹是否存在在 train.jsonl 文件
    train_dataset = os.path.join(dst_dir, "train.jsonl")
    if not os.path.exists(train_dataset):
        logging.info("train dataset not exist. creating.")
        file = open(train_dataset, 'w')
        file.close()
        logging.info("train dataset created.")    
    
    # summarizer 策略：warmup 阶段固定 default_perf（数据量过小无法形成稳态窗口），全量阶段用配置值
    logging.info(f"summarizer strategy: warmup -> {WARMUP_SUMMARIZER} (fixed), full dataset -> {DEFAULT_PERFORMANCE_TEST} (config)")

    pd_monitor = PDMetricsMonitor.from_config()
    if pd_monitor.enabled:
        if pd_monitor.mode == "colocated":
            logging.info(f"混部 runtime metrics monitor enabled (M_LISTEN_SERVER 优先): {pd_monitor.endpoints}")
            if pd_monitor.ignored_pd:
                logging.info(f"已忽略 P/D 监听 {pd_monitor.ignored_pd}（M_LISTEN_SERVER 已填写，优先走混部）")
        else:
            logging.info(f"PD runtime metrics monitor enabled: {pd_monitor.endpoints}")
        logging.info(f"runtime metrics live file: {pd_monitor.live_file_path}")
        logging.info(f"实时监控(文件覆盖刷新，不刷屏): tail -F {pd_monitor.live_file_path}")
        pd_monitor.start()
    else:
        logging.info("runtime metrics monitor disabled (M_LISTEN_SERVER 与 P_LISTEN_SERVER/D_LISTEN_SERVER 均未配置)")

    # PD 请求五阶段耗时分析（KV_STAGE_LOG_P/D 配置后启用；仅 PD 分离形态）
    kv_analyzer = KVStageAnalyzer.from_config()
    if kv_analyzer.enabled and pd_monitor.enabled and pd_monitor.mode == "colocated":
        logging.warning("[kv_stage] 混部形态不支持五阶段分析，已忽略 KV_STAGE_LOG_P/D 配置")
        kv_analyzer.enabled = False
    if kv_analyzer.enabled and not pd_monitor.enabled:
        logging.warning("[kv_stage] 运行时监控未启用（*_LISTEN_SERVER 未配置），五阶段结果仅输出打屏与 CSV，不生成图 3")

    def run_aisbench(cmd, phase):
        logging.info(f"test start ({phase}), use command: {cmd}")
        if pd_monitor.enabled:
            pd_monitor.set_phase(phase)
            logging.info(f"runtime metrics phase -> {phase}")
            if is_full_dataset_phase(phase):
                pd_monitor.capture_full_dataset_baseline()
                logging.info("Captured full_dataset prefix-cache baseline (hits/queries), warmup excluded")
        if kv_analyzer.enabled:
            kv_analyzer.start()
        # 独立进程组：超时时连同 `sh -c "ais_bench | tee"` 的子进程树一起终止，
        # 否则只杀外层 shell，压测仍在后台打服务
        proc = subprocess.Popen(cmd, shell=True, start_new_session=True)
        try:
            return_code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (AttributeError, OSError):
                proc.kill()
            proc.wait()
            logging.error(f"阶段 {phase} 超时（{timeout}s），已终止进程组")
            raise SystemExit(f"测试阶段 {phase} 超时（{timeout}s），请检查服务状态或增大 --timeout")
        if return_code != 0:
            if "--summarizer stable_stage" in cmd:
                logging.error(
                    "[稳态测试失败] ais_bench 未找到稳态窗口（Can not find a stable stage），"
                    "常见原因：数据量太小或并发未能两次打满。"
                    "可增大 --data_num，或将 config.py 的 DEFAULT_PERFORMANCE_TEST 改为 default_perf。"
                )
            raise subprocess.CalledProcessError(return_code, cmd)
        if pd_monitor.enabled:
            plot_dir = resolve_plot_output_dir(OUTPUT_DIR, api_test_abbr)
            plot_now = find_aisbench_plot_html(plot_dir)
            pd_monitor.remember_aisbench_plot(plot_now, plot_label_from_phase(phase))
            logging.info(f"recorded ais_bench plot for phase {phase}: {plot_now}")
        if pd_monitor.enabled and is_full_dataset_phase(phase):
            snapshots = pd_monitor.capture_full_dataset_end()
            for item in snapshots:
                rate = item.get("prefix_hit_rate")
                rate_text = f"{rate:.2f}%" if rate is not None else "N/A"
                logging.info(
                    f"full_dataset prefix hit rate {role_label(item.get('role'))} {item.get('endpoint')}: "
                    f"Δhits={item.get('prefix_cache_hits_delta')} "
                    f"Δqueries={item.get('prefix_cache_queries_delta')} "
                    f"rate={rate_text}"
                )
        if kv_analyzer.enabled:
            kv_analyzer.collect(phase)

    try:
        # 执行命令
        per_round_saved = False
        if dataset_type == "prefix_cache":
            # 前缀数据集测试
            if prefix_test:
                if not POD_INFO:
                    pod_info = [HOST_IP+":"+HOST_PORT]
                else:
                    pod_info = POD_INFO
                logging.info(f"pod_info: {pod_info}")
                # 首次拉取指标前整体校验一次 POD_INFO，格式非法则立即失败退出
                for _pod in pod_info:
                    try:
                        parse_endpoint(_pod)
                    except ValueError as error:
                        logging.error(str(error))
                        sys.exit(1)

                logging.info(f"[开始] 前缀数据集测试")
                # 前缀测试: request rate 固定为 0
                modify_aisbench_api(str(dp), "1", request_rate_param="0")
                dst_file = generate_test_dataset(src_file_prefix, dst_dir)

                # 命中率计算
                query_tokens, query_tokens_external, hit_tokens, hit_tokens_external = get_pod_metrics_info(pod_info)

                run_aisbench(generate_aisbench_command(WARMUP_SUMMARIZER), "prefix_warmup")
                logging.info(f"[完成] 前缀数据集测试完成，结果保存在aisbench_result.csv")

                query_tokens_new, query_tokens_external_new, hit_tokens_new, hit_tokens_external_new = get_pod_metrics_info(pod_info)
                cal_prefix_hit_info(query_tokens, query_tokens_external, hit_tokens, hit_tokens_external,query_tokens_new,
                                    query_tokens_external_new, hit_tokens_new, hit_tokens_external_new)

                # 保存前缀测试结果（warmup 阶段固定 default_perf 口径；accuracy 模式如实标记）
                save_result(0, npu_num, summarizer="accuracy" if test_accuracy else WARMUP_SUMMARIZER)
                logging.info(f"[开始] 全量数据集测试")
                # 命中率计算
                query_tokens, query_tokens_external, hit_tokens, hit_tokens_external = get_pod_metrics_info(pod_info)

                modify_aisbench_api(concurrency,str(output_len))
                dst_file = generate_test_dataset(src_file_data, dst_dir)
                # 执行测试命令
                run_aisbench(generate_aisbench_command(DEFAULT_PERFORMANCE_TEST), "full_dataset")
                logging.info(f"[完成] 全量数据集测试完成，结果保存在aisbench_result.csv")

                query_tokens_new, query_tokens_external_new, hit_tokens_new, hit_tokens_external_new = get_pod_metrics_info(pod_info)
                cal_prefix_hit_info(query_tokens, query_tokens_external, hit_tokens, hit_tokens_external, query_tokens_new,
                                    query_tokens_external_new, hit_tokens_new, hit_tokens_external_new)

            else:
                logging.info(f"[开始] 全量数据集测试")
                modify_aisbench_api(concurrency,str(output_len))
                dst_file = generate_test_dataset(src_file_data, dst_dir)
                run_aisbench(generate_aisbench_command(DEFAULT_PERFORMANCE_TEST), "full_dataset")
                logging.info(f"[完成] 全量数据集测试完成，结果保存在aisbench_result.csv")

        else:
            logging.info(f"[开始] 全量数据集测试")
            modify_aisbench_api(concurrency,str(output_len))
            dst_file = generate_test_dataset(src_file_data, dst_dir)
            if test_times > 1:
                per_round_saved = True
                round_summarizer = DEFAULT_PERFORMANCE_TEST
                for test_time in range(test_times):
                    logging.info(f"Execution rounds: {test_time + 1}")
                    run_aisbench(generate_aisbench_command(DEFAULT_PERFORMANCE_TEST), f"full_dataset_round_{test_time + 1}")
                    # 每轮单独落一行（round 列记录轮次），避免只保留最后一轮
                    save_result(request_rate, npu_num, extra_metrics=None,
                                summarizer=round_summarizer, round_index=test_time + 1)
            else:
                run_aisbench(generate_aisbench_command(DEFAULT_PERFORMANCE_TEST), "full_dataset")
            logging.info(f"[完成] 全量数据集测试完成，结果保存在aisbench_result.csv")

        # 保存结果（accuracy 模式无 perf summarizer，如实标记）
        extra_metrics = None
        if pd_monitor.enabled:
            extra_metrics = {"Prefix Hit Rate": pd_monitor.format_hit_rate_for_csv()}
            pd_monitor.print_hit_rate_table()
        if kv_analyzer.enabled and kv_analyzer.records:
            kv_analyzer.log_summary()
            kv_metrics = kv_analyzer.csv_extra_metrics()
            if extra_metrics:
                extra_metrics.update(kv_metrics)
            else:
                extra_metrics = kv_metrics
            kv_perf_dir = resolve_plot_output_dir(OUTPUT_DIR, api_test_abbr)
            kv_analyzer.align_with_aisbench_details(kv_perf_dir)
            kv_analyzer.write_details_csv(kv_perf_dir)
            if pd_monitor.enabled:
                pd_monitor.set_kv_stage_data(kv_analyzer.build_bar_traces())
        # 多轮模式已在每轮落行：汇总类指标（Prefix Hit Rate / 五阶段均值）属于全量聚合，
        # 追加写入最后一轮行，避免被 `per_round_saved` 跳过而静默丢失
        if per_round_saved:
            if extra_metrics:
                merged = merge_extra_metrics_into_csv("aisbench_result.csv", extra_metrics, test_times)
                if merged:
                    logging.info(
                        f"多轮模式：已将聚合指标 {sorted(extra_metrics)} 写入第 {test_times} 轮行"
                        "（Prefix Hit Rate 仅代表最后一轮全量统计，五阶段为全程聚合）"
                    )
                else:
                    logging.error("多轮模式：聚合指标写入 CSV 失败，请检查 aisbench_result.csv")
        else:
            save_result(request_rate, npu_num, extra_metrics,
                        summarizer="accuracy" if test_accuracy else DEFAULT_PERFORMANCE_TEST)
    finally:
        if pd_monitor.enabled:
            pd_monitor.stop()
            perf_dir = resolve_plot_output_dir(OUTPUT_DIR, api_test_abbr)
            exported = pd_monitor.export(perf_dir)
            if exported:
                logging.info(f"runtime metrics HTML: {exported.get('html')}")
                logging.info(f"runtime metrics JSONL: {exported.get('jsonl')}")
