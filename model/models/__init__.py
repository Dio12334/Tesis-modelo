"""Model registry and detection model wrappers."""

from model.models.registry import BaseDetector, ModelRegistry
from model.models.mobilenetv4_ssd import MobileNetV4Detector
from model.models.rt_detr_wrapper import RT_DETR_Detector
from model.models.ssd_mobilenet import SSDMobileNetV3
from model.models.yolo26_wrapper import YOLO26Detector
from model.models.yolov6_wrapper import YOLOv6Detector

__all__ = [
    "BaseDetector",
    "ModelRegistry",
    "MobileNetV4Detector",
    "RT_DETR_Detector",
    "SSDMobileNetV3",
    "YOLO26Detector",
    "YOLOv6Detector",
]
