"""Equirectangular (360°) -> Perspective projection.

Copia EXACTA del renderer usado por la herramienta de anotación. Los bounding boxes están
normalizados respecto de esta vista perspectiva, así que NO se deben modificar los defaults
(fov 110x120, salida 1920x1080) o las cajas dejarán de coincidir con la imagen.
"""
from __future__ import annotations

import cv2
import numpy as np


def _build_rotation_matrix(yaw_rad: float, pitch_rad: float) -> np.ndarray:
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    R_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    R_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    return R_yaw @ R_pitch


def _build_remap_maps(W_src, H_src, W_out, H_out, yaw_rad, pitch_rad, fov_h_rad, fov_v_rad):
    u = np.arange(W_out, dtype=np.float32)
    v = np.arange(H_out, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    x_cam = (uu / W_out - 0.5) * 2.0 * np.tan(fov_h_rad * 0.5)
    y_cam = -(vv / H_out - 0.5) * 2.0 * np.tan(fov_v_rad * 0.5)
    z_cam = np.ones_like(x_cam)
    rays = np.stack([x_cam, y_cam, z_cam], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    R = _build_rotation_matrix(yaw_rad, pitch_rad)
    world = (R @ rays.reshape(-1, 3).astype(np.float64).T).T
    x_w, y_w, z_w = world[:, 0], world[:, 1], world[:, 2]
    theta = np.arctan2(x_w, z_w)
    phi = np.arcsin(np.clip(y_w, -1.0, 1.0))
    map_x = ((theta / np.pi + 1.0) * 0.5 * W_src).astype(np.float32).reshape(H_out, W_out)
    map_y = ((-phi / (np.pi * 0.5) + 1.0) * 0.5 * H_src).astype(np.float32).reshape(H_out, W_out)
    return map_x, map_y


def _decode_bytes(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image bytes")
    return img


def render_view_bytes(image_bytes: bytes, yaw: float, pitch: float = -25.0,
                      fov_h: float = 110.0, fov_v: float = 120.0,
                      out_w: int = 1920, out_h: int = 1080) -> bytes:
    """360° bytes -> perspective JPEG bytes. Los defaults DEBEN coincidir con los de la
    herramienta de anotación (fov 110x120, salida 1920x1080)."""
    src = _decode_bytes(image_bytes)
    H_src, W_src = src.shape[:2]
    map_x, map_y = _build_remap_maps(
        W_src, H_src, out_w, out_h,
        np.deg2rad(yaw), np.deg2rad(pitch), np.deg2rad(fov_h), np.deg2rad(fov_v),
    )
    result = cv2.remap(src.astype(np.float32), map_x, map_y,
                       interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    result = np.clip(result, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("Failed to encode perspective view as JPEG")
    return buf.tobytes()
