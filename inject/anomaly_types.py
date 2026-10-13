from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class AnomalyClass:
    id: int
    name: str
    operators: List[int]
    intensities: List[float]
    length_ratio: float


@dataclass
class AnomalyInstance:
    class_id: int
    class_name: str
    operators: List[int]
    intensities: List[float]
    start: int
    end: int
    length: int
    mask: np.ndarray
    affected_dims: List[int]