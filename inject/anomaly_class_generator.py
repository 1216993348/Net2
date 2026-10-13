"""
异常类生成器 - 6个算子，每个是一个类
"""

import numpy as np
from typing import Dict
from inject.anomaly_types import AnomalyClass
from inject.operators import OPERATOR_MAP


OPERATOR_WEIGHTS = {
    "Spike": 0.18,
    "Drift": 0.18,
    "Shift": 0.18,
    "Period": 0.18,
    "Cascade": 0.10,
    "Missing": 0.18,
}


class AnomalyClassGenerator:
    def __init__(self):
        self.op_name_to_id = {}
        for op_id, op_class in OPERATOR_MAP.items():
            self.op_name_to_id[op_class.__name__] = op_id
        self.op_weights = OPERATOR_WEIGHTS

    def generate(self) -> Dict[int, AnomalyClass]:
        classes = {}
        for op_id, op_class in OPERATOR_MAP.items():
            op_name = op_class.__name__
            classes[op_id] = AnomalyClass(
                id=op_id,
                name=op_name,
                operators=[op_id],
                intensities=[0.0],
                length_ratio=0.05,
            )
        return classes

    def sample_class_id(self) -> int:
        op_names = list(self.op_weights.keys())
        weights = list(self.op_weights.values())
        probs = np.array(weights) / sum(weights)
        op_name = np.random.choice(op_names, p=probs)
        return self.op_name_to_id[op_name]