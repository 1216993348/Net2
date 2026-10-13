import numpy as np
from inject.operators import OPERATOR_MAP

class MaskedUniversalInjector:
    """
    轻量级通用异常注入器 (Masked 版本)
    完全随机混合多种异常算子，稳定输出 injected_data 和 mask，适用于训练时的对比增强。
    """
    def __init__(self):
        # 获取六大算子类列表[cite: 11]
        self.operators = list(OPERATOR_MAP.values())

    def _sample_intensity(self):
        # 随机采样异常强度
        return np.random.uniform(0.5, 2.5)

    def inject(self, ts_data: np.ndarray):
        """
        注入异常
        返回:
            injected: 注入异常后的序列 [L, C]
            mask: 异常位置标签 [L] (1 为异常, 0 为正常)
        """
        T, D = ts_data.shape
        injected = ts_data.copy()
        mask = np.zeros(T, dtype=np.float32)

        # 1. 决定异常的长度 (总长度的 5% ~ 20%)
        min_len = max(5, int(T * 0.05))
        max_len = max(10, int(T * 0.20))
        length = np.random.randint(min_len, max_len + 1)

        # 2. 决定异常的起始位置
        start = np.random.randint(0, max(1, T - length))
        end = start + length

        # 3. 决定受影响的变量/维度 (随机影响 1 到 D//2 个通道，避免全通道崩溃)
        max_dims = max(1, D // 2)
        num_dims = np.random.randint(1, max_dims + 1)
        affected_dims = np.random.choice(D, size=num_dims, replace=False)

        # 4. 随机选取 1~2 个算子进行组合 (例如 Spike 叠加 Shift)
        num_ops = np.random.choice([1, 2], p=[0.7, 0.3])
        selected_ops = np.random.choice(self.operators, size=num_ops, replace=False)

        # 5. 应用算子进行篡改
        for op in selected_ops:
            intensity = self._sample_intensity()
            for dim in affected_dims:
                try:
                    # 调用算子逻辑[cite: 11]
                    injected = op.apply(injected, dim, intensity, start, end)
                except Exception:
                    # 某个算子应用失败时静默跳过，保证训练不中断
                    pass

        # 6. 生成准确的时间步 Mask
        mask[start:end] = 1.0

        return injected, mask