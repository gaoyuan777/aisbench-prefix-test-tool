import json
import random
import os

# 模块级语料缓存：按绝对路径缓存 jsonl 原始行，避免每次实例化/每次挑选都重新读取与计数
_CORPUS_CACHE = {}


def _load_corpus(jsonl_file):
    """读取并缓存 jsonl 文件的所有行；同一文件仅读取一次"""
    abs_path = os.path.abspath(jsonl_file)
    if abs_path not in _CORPUS_CACHE:
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"数据集文件不存在: {jsonl_file}")
        with open(abs_path, 'r', encoding='utf-8') as f:
            _CORPUS_CACHE[abs_path] = f.readlines()
    return _CORPUS_CACHE[abs_path]


class DataPicker:
    def __init__(self, jsonl_file, record_file="picked_ids.txt", prefix_flag=0, rng=None):
        """
        初始化挑选器

        Args:
            jsonl_file: GSM8K数据集文件路径
            record_file: 记录已挑选数据编号的文件路径
            prefix_flag: 1 表示不重复挑选模式
            rng: 显式 random.Random 实例，用于可复现挑选；缺省时每个实例自建
        """
        self.jsonl_file = jsonl_file
        self.record_file = record_file
        self._lines = _load_corpus(jsonl_file)
        self.total_lines = len(self._lines)
        self.picked_ids = self._load_picked_ids()
        self.prefix_flag = prefix_flag
        self.rng = rng if rng is not None else random.Random()

    def _load_picked_ids(self):
        """加载已挑选的编号记录"""
        if os.path.exists(self.record_file):
            with open(self.record_file, 'r', encoding='utf-8') as f:
                return set(int(line.strip()) for line in f if line.strip())
        return set()

    def _save_picked_ids(self):
        """保存已挑选的编号记录"""
        with open(self.record_file, 'w', encoding='utf-8') as f:
            for pid in sorted(self.picked_ids):
                f.write(f"{pid}\n")

    def get_unpicked_ids(self):
        """获取所有未被挑选过的编号"""
        all_ids = set(range(self.total_lines))
        return list(all_ids - self.picked_ids)

    def pick_one(self):
        """
        随机挑选一条数据

        Returns:
            挑选的数据内容，如果没有未挑选的数据则返回None
        """

        # 获取可选的编号列表
        if self.prefix_flag == 1:
            # 不重复模式：从未挑选的编号中选择
            available_ids = self.get_unpicked_ids()
            if not available_ids:
                return None
            # 随机选择一个未挑选的编号
            selected_id = self.rng.choice(available_ids)
            # 记录已挑选的编号
            self.picked_ids.add(selected_id)
            self._save_picked_ids()
        else:
            # 可重复模式：从所有编号中随机选择，不记录编号
            if self.total_lines <= 0:
                return None
            selected_id = self.rng.randint(0, self.total_lines - 1)

        # 越界保护（文件可能已被外部修改或索引越界）
        if selected_id < 0 or selected_id >= self.total_lines:
            raise ValueError(f"数据集文件行数变化或索引越界: {selected_id} >= {self.total_lines}")

        # 从缓存的语料中读取对应编号的数据
        data = json.loads(self._lines[selected_id])

        return data['question']
