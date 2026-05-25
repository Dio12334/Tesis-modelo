"""Model registry and detection model wrappers."""

from model.models.registry import BaseDetector, ModelRegistry
from model.models.ssd_mobilenet import SSDMobileNetV3
from model.models.yolov6_wrapper import YOLOv6Detector

__all__ = ["BaseDetector", "ModelRegistry", "SSDMobileNetV3", "YOLOv6Detector"]
