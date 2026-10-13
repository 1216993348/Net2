"""
通用异常注入器
"""

import numpy as np
from typing import List, Tuple, Optional

from inject.anomaly_types import AnomalyClass, AnomalyInstance
from inject.anomaly_class_generator import AnomalyClassGenerator
from inject.operators import OPERATOR_MAP


class UniversalAnomalyInjector:
    def __init__(self):
        self.generator = AnomalyClassGenerator()
        self.anomaly_classes = self.generator.generate()
        self.num_classes = len(self.anomaly_classes)
        print(f"  Created {self.num_classes} anomaly classes")

    def _sample_intensity(self) -> float:
        return np.exp(np.random.uniform(np.log(0.3), np.log(1.5)))

    def _sample_length(self, T: int) -> int:
        min_len = max(5, int(T * 0.02))
        max_len = int(T * 0.15)
        log_min, log_max = np.log(min_len), np.log(max_len)
        return int(np.exp(np.random.uniform(log_min, log_max)))

    def _sample_affected_dims(self, n_dims: int) -> List[int]:
        if np.random.rand() < 0.5:
            return [np.random.randint(0, n_dims)]
        else:
            max_k = min(n_dims, int(n_dims * 0.2))
            k = np.random.randint(2, max_k + 1) if max_k >= 2 else 1
            return np.random.choice(n_dims, size=k, replace=False).tolist()

    def _is_injection_effective(self, original: np.ndarray, injected: np.ndarray) -> bool:
        """判断注入是否有效"""
        # 1. 检查是否完全相同
        if np.allclose(original, injected, atol=1e-6):
            return False

        # 2. 检查均值变化
        if abs(np.mean(injected) - np.mean(original)) > 1e-4:
            return True

        # 3. 检查标准差变化
        if abs(np.std(injected) - np.std(original)) > 1e-4:
            return True

        # 4. 检查最大变化
        if np.max(np.abs(injected - original)) > 1e-3:
            return True

        return False

    def inject(self, ts_data: np.ndarray, max_attempts: int = 3) -> Tuple[np.ndarray, Optional[AnomalyInstance]]:
        T, D = ts_data.shape

        for attempt in range(max_attempts):
            class_id = self.generator.sample_class_id()
            anomaly_class = self.anomaly_classes[class_id]

            length = self._sample_length(T)
            start_idx = np.random.randint(0, max(1, T - length))
            end_idx = start_idx + length

            affected_dims = self._sample_affected_dims(D)
            intensities = [self._sample_intensity() for _ in anomaly_class.operators]

            original_segment = ts_data[start_idx:end_idx].copy()
            test_data = ts_data.copy()

            for op_id, intensity in zip(anomaly_class.operators, intensities):
                op = OPERATOR_MAP.get(op_id)
                if op is None:
                    continue
                for dim in affected_dims:
                    test_data = op.apply(test_data, dim, intensity, start_idx, end_idx)

            valid = True
            for dim in affected_dims:
                original_col = original_segment[:, dim]
                injected_col = test_data[start_idx:end_idx, dim]
                if not self._is_injection_effective(original_col, injected_col):
                    valid = False
                    break

            if valid:
                ts_data = test_data
                mask = np.zeros(T, dtype=np.float32)
                mask[start_idx:end_idx] = 1.0

                instance = AnomalyInstance(
                    class_id=class_id,
                    class_name=anomaly_class.name,
                    operators=anomaly_class.operators,
                    intensities=intensities,
                    start=start_idx,
                    end=end_idx,
                    length=length,
                    mask=mask,
                    affected_dims=affected_dims,
                )
                return ts_data, instance

        return ts_data, None

    def inject_batch(self, batch_data: np.ndarray, anomaly_ratio: float = 0.3) -> Tuple[np.ndarray, List[Optional[AnomalyInstance]]]:
        B = batch_data.shape[0]
        augmented = batch_data.copy()
        instances = []

        n_anomaly = int(B * anomaly_ratio)
        anomaly_indices = np.random.choice(B, size=n_anomaly, replace=False)

        for idx in range(B):
            if idx in anomaly_indices:
                augmented[idx], instance = self.inject(batch_data[idx])
                instances.append(instance)
            else:
                instances.append(None)

        return augmented, instances