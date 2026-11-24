"""从 /metrics 抓取 RAW 累计计数器（prefix_cache_queries/hits_total）；
命中率公式在 aisbench_test.cal_prefix_hit_info，文件名系历史遗留，勿改名。"""
import logging
import re
import requests

logging.getLogger().setLevel(logging.INFO)


def _fetch_metrics(ip_address, port):
    """获取 /metrics 接口文本"""
    url = f"http://{ip_address}:{port}/metrics"
    resp = requests.get(url, proxies={"http": None, "https": None}, timeout=30)
    resp.raise_for_status()
    return resp.text


def _get_prefix_cache_metrics(ip_address, port, metric_type):
    """
    通用的前缀缓存指标获取函数

    Args:
        ip_address: IP 地址
        port: 端口号
        metric_type: 指标类型，'queries' 或 'hits'

    Returns:
        tuple: (normal_stats, external_stats) 两个字典，key 为 engine_id(int)，value 为指标值
    """
    try:
        text = _fetch_metrics(ip_address, port)
        lines = [l for l in text.split('\n')
                 if 'model_name' in l and f'prefix_cache_{metric_type}_total' in l]
        normal_stats = {}
        external_stats = {}

        for line in lines:
            # 提取engine值
            engine_match = re.search(r'engine="(\d+)"', line)
            if not engine_match:
                continue

            engine = engine_match.group(1)

            # 取标签块 '}' 之后的第一个数字字段，兼容 OpenMetrics exemplar / 时间戳后缀
            brace_idx = line.rfind('}')
            tail = line[brace_idx + 1:] if brace_idx != -1 else line
            value = None
            for tok in tail.split():
                try:
                    value = float(tok)
                    break
                except ValueError:
                    continue
            if value is None:
                # 兜底：仍按原逻辑取最后一个字段
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    value = float(parts[-1])
                except ValueError:
                    continue
            if value.is_integer():
                value = int(value)

            # 根据指标名称分类；同一 engine 出现在多行时对计数器求和而非覆盖
            if f'external_prefix_cache_{metric_type}_total' in line:
                external_stats[int(engine)] = external_stats.get(int(engine), 0) + value
            elif f'vllm:prefix_cache_{metric_type}_total' in line:
                normal_stats[int(engine)] = normal_stats.get(int(engine), 0) + value

        logging.info(f"{metric_type}: normal={normal_stats}, external={external_stats}")
        return normal_stats, external_stats
    except Exception as e:
        logging.error(f"获取{metric_type}指标失败: {e}")
        return {}, {}


def get_prefix_queries_total(ip_address, port):
    """
    获取查询token总数,返回{engine:tokens}
    """
    return _get_prefix_cache_metrics(ip_address, port, 'queries')


def get_prefix_hits_total(ip_address, port):
    """
    获取命中token总数,返回{engine:tokens}
    """
    return _get_prefix_cache_metrics(ip_address, port, 'hits')