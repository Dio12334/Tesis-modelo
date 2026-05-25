"""Dataset interfaces and implementations."""

from model.datasets.base import Annotation, BaseDataset, BoundingBox
from model.datasets.rdd2022 import RDD2022Dataset
from model.datasets.target_mapper import TargetMapper

__all__ = ["Annotation", "BaseDataset", "BoundingBox", "RDD2022Dataset", "TargetMapper"]
