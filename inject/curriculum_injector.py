"""
课程学习注入器 - 细粒度渐进式异常注入
"""

import numpy as np
from typing import Tuple, List, Dict, Any
from inject.operators import OPERATOR_MAP


def get_operator_by_name(op_name: str):
    """根据算子名称获取算子类"""
    for op_id, op_class in OPERATOR_MAP.items():
        if op_class.__name__ == op_name:
            return op_class
    raise ValueError(f"Unknown operator: {op_name}")


class CurriculumInjector:
    """
    课程学习注入器 - 细粒度渐进式难度增加

    设计原则：
    - 每次只引入一个"新东西"
    - 从简单到复杂逐步推进
    - 先学形状，再学位置，再学维度，再学叠加
    """

    def __init__(self, total_epochs: int = 50):
        self.total_epochs = total_epochs

        # 细粒度阶段配置
        # 每个阶段: (epoch_start, epoch_end, ops, position, n_dims, n_ops, desc)
        self.stages = [
            # ========== Spike 基础 ==========
            (0, 0, ["Spike"], "random", 1, 1, "Spike固定位置"),
            # ========== +Shift ==========
            (1, 1, ["Spike", "Shift"], "random", 1, 1, "+Shift"),

            # ========== +Drift ==========
            (2, 5, ["Spike", "Shift", "Drift"], "random", 1, 1, "+Drift"),

            # ========== +Missing ==========
            (11, 13, ["Spike", "Shift", "Drift", "Missing"], "random", 1, 1, "+Missing"),

            # ========== +Period ==========
            (14, 16, ["Spike", "Shift", "Drift", "Missing", "Period"], "random", 1, 1, "+Period"),

            # ========== +Cascade ==========
            (17, 19, ["Spike", "Shift", "Drift", "Missing", "Period", "Cascade"], "random", 1, 1, "+Cascade"),

            # ========== 回顾所有算子 ==========
            (15, 16, ["Spike", "Shift", "Drift", "Missing", "Period", "Cascade"], "random", 1, 1, "回顾所有算子"),

            # ========== 多变量（首次） ==========
            (17, 18, ["Spike", "Shift", "Drift", "Missing", "Period", "Cascade"], "random", 2, 1, "多变量(2个)"),

            # ========== 多变量变长 ==========
            (19, 20, ["Spike", "Shift", "Drift", "Missing", "Period", "Cascade"], "random", (1, 2), 1, "多变量(1-2个)"),

            # ========== 算子叠加 ==========
            (21, 22, ["Spike", "Shift", "Drift", "Missing", "Period", "Cascade"], "random", (1, 2), (1, 2), "算子叠加(1-2个)"),

            # ========== 全复杂度 ==========
            (23, 50, ["Spike", "Shift", "Drift", "Missing", "Period", "Cascade"], "random", (1, 3), (1, 3), "全复杂度"),
        ]

    def get_stage(self, epoch: int) -> Dict[str, Any]:
        """根据 epoch 返回当前阶段的配置"""
        for start, end, ops, position, n_dims, n_ops, desc in self.stages:
            if start <= epoch <= end:
                return {
                    'ops': ops,
                    'position': position,
                    'n_dims': n_dims,
                    'n_ops': n_ops,
                    'desc': desc,
                    'epoch_range': (start, end)
                }
        # 默认返回最后一个阶段
        return {
            'ops': ["Spike", "Shift", "Drift", "Missing", "Period", "Cascade"],
            'position': 'random',
            'n_dims': (1, 3),
            'n_ops': (1, 3),
            'desc': '全复杂度',
            'epoch_range': (23, 50)
        }

    def _sample_ops(self, ops_list: List[str], n_ops_config) -> List[str]:
        """采样算子"""
        if isinstance(n_ops_config, tuple):
            k = np.random.randint(n_ops_config[0], n_ops_config[1] + 1)
        else:
            k = n_ops_config
        k = min(k, len(ops_list))
        return np.random.choice(ops_list, size=k, replace=False).tolist()

    def _sample_affected_dims(self, D: int, n_dims_config) -> List[int]:
        """采样受影响的维度"""
        if isinstance(n_dims_config, tuple):
            n_dims = np.random.randint(n_dims_config[0], n_dims_config[1] + 1)
        else:
            n_dims = n_dims_config
        n_dims = min(n_dims, D)
        return np.random.choice(D, size=n_dims, replace=False).tolist()

    def _sample_length(self, epoch: int, T: int) -> int:
        """根据 epoch 渐进增加异常长度"""
        if epoch < 5:
            length = np.random.randint(8, 15)
        elif epoch < 10:
            length = np.random.randint(10, 20)
        elif epoch < 15:
            length = np.random.randint(12, 25)
        elif epoch < 20:
            length = np.random.randint(15, 30)
        else:
            length = np.random.randint(20, 40)
        return min(length, T // 2)

    def _sample_intensity(self, epoch: int) -> float:
        """根据 epoch 渐进增加异常强度"""
        if epoch < 5:
            return np.random.uniform(0.5, 0.8)
        elif epoch < 10:
            return np.random.uniform(0.6, 1.0)
        elif epoch < 15:
            return np.random.uniform(0.7, 1.2)
        elif epoch < 20:
            return np.random.uniform(0.8, 1.4)
        else:
            return np.random.uniform(1.0, 1.8)

    def _get_position(self, position_config: str, T: int, length: int) -> int:
        """获取异常起始位置"""
        if position_config == 'fixed':
            return T // 2 - length // 2
        else:
            return np.random.randint(0, max(1, T - length))

    def inject(self, ts_data: np.ndarray, epoch: int) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
        """
        注入异常

        Args:
            ts_data: (T, D) 原始时间序列
            epoch: 当前 epoch

        Returns:
            injected_data: (T, D) 注入后的数据
            mask: (T,) 异常掩码
            used_ops: 使用的算子列表
            affected_dims: 受影响的维度列表
        """
        T, D = ts_data.shape
        config = self.get_stage(epoch)

        ops_list = config['ops']
        position = config['position']
        n_dims_config = config['n_dims']
        n_ops_config = config['n_ops']

        # 1. 选择算子
        selected_ops = self._sample_ops(ops_list, n_ops_config)

        # 2. 采样长度
        length = self._sample_length(epoch, T)

        # 3. 采样位置
        start = self._get_position(position, T, length)
        end = min(start + length, T)
        length = end - start

        if length <= 0:
            return ts_data.copy(), np.zeros(T, dtype=np.float32), [], []

        # 4. 采样维度
        affected_dims = self._sample_affected_dims(D, n_dims_config)

        # 5. 采样强度
        intensity = self._sample_intensity(epoch)

        # 6. 注入异常
        injected = ts_data.copy()

        for op_name in selected_ops:
            op = get_operator_by_name(op_name)
            for dim in affected_dims:
                try:
                    injected = op.apply(injected, dim, intensity, start, end)
                except Exception as e:
                    # 算子失败时静默跳过（避免训练中断）
                    continue

        # 7. 构建掩码
        mask = np.zeros(T, dtype=np.float32)
        mask[start:end] = 1.0

        return injected, mask, selected_ops, affected_dims

    def get_stage_info(self, epoch: int) -> str:
        """获取当前阶段的描述信息（用于日志）"""
        config = self.get_stage(epoch)
        return f"[Stage: {config['desc']}]"