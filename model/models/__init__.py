"""Model registry and detection model wrappers."""

from model.models.registry import BaseDetector, ModelRegistry
from model.models.ssd_mobilenet import SSDMobileNetV3
from model.models.yolo26_wrapper import YOLO26Detector
from model.models.yolov6_wrapper import YOLOv6Detector

__all__ = [
    "BaseDetector",
    "ModelRegistry",
    "SSDMobileNetV3",
    "YOLO26Detector",
    "YOLOv6Detector",
]
