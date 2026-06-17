"""Unit tests for the SSDLite + MobileNetV3-Large detector wrapper.

Covers:
* Config validation (missing keys, out-of-range values, paired LR).
* CONFIG_SCHEMA contract used by ``ModelRegistry``.
* Registry-driven construction.
* Discriminative-LR parameter grouping.
* Freeze-schedule hook (``on_epoch_start``) and optimizer-rebuild flag.
* ``forward()`` contract (pixel-xyxy boxes, 0-indexed labels).
* ``train_step()`` label-shift (+1 for background) and empty-batch handling.
* Checkpoint round-trip preserves config + state.

Tests that require an actual torchvision SSD build (i.e. heavy: downloads
weights on first run) are gated behind ``requires_torchvision`` so the
schema / validation half of the suite runs without torch installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from model.exceptions import ConfigurationError
from model.models.registry import ModelRegistry

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import torchvision  # noqa: F401
    HAS_TORCHVISION = HAS_TORCH
except ImportError:
    HAS_TORCHVISION = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
requires_torchvision = pytest.mark.skipif(
    not HAS_TORCHVISION, reason="torchvision not installed"
)


# ---------------------------------------------------------------------------
# Schema / validation tests (no torch needed)
# ---------------------------------------------------------------------------


def _import_module():
    """Import lazily so the file can be collected without torch.

    Importing ``model.models.ssd_mobilenet`` would pull in torchvision
    at module import time. We tolerate that for tests that already require
    torchvision; for pure-schema tests we still import and skip on failure.
    """
    try:
        import model.models.ssd_mobilenet as mod
        return mod
    except ImportError as e:
        pytest.skip(f"ssd_mobilenet module unavailable: {e}")


@requires_torch
def test_config_schema_required_keys():
    mod = _import_module()
    schema = mod.CONFIG_SCHEMA
    assert schema["input_size"]["required"] is True
    assert schema["num_classes"]["required"] is True
    # Optional keys should be marked as not required.
    for k in (
        "pretrained_backbone",
        "trainable_backbone_layers",
        "confidence_threshold",
        "nms_threshold",
        "backbone_lr",
        "head_lr",
        "freeze_backbone_epochs",
    ):
        assert k in schema, f"Schema missing optional key {k!r}"
        assert schema[k]["required"] is False


@requires_torch
def test_get_config_schema_callable_via_object_new():
    """ModelRegistry._get_schema calls get_config_schema on object.__new__.

    The method must therefore work without ``__init__`` having run.
    """
    mod = _import_module()
    instance = object.__new__(mod.SSDMobileNetV3)
    schema = instance.get_config_schema()
    assert schema is mod.CONFIG_SCHEMA


@requires_torchvision
def test_registry_validates_missing_params():
    mod = _import_module()  # ensures registration runs
    with pytest.raises(ConfigurationError) as exc:
        ModelRegistry.create("ssd_mobilenetv3", {})
    violations = exc.value.violations
    joined = " ".join(violations)
    assert "input_size" in joined
    assert "num_classes" in joined


@requires_torchvision
@pytest.mark.parametrize("bad_size", [0, 224, 416, 512, 1024, -1])
def test_invalid_input_size_raises(bad_size):
    mod = _import_module()
    with pytest.raises(ConfigurationError) as exc:
        mod.SSDMobileNetV3({"input_size": bad_size, "num_classes": 5})
    assert any("input_size" in v for v in exc.value.violations)


@requires_torchvision
@pytest.mark.parametrize("bad_nc", [0, -1, 1001, "five"])
def test_invalid_num_classes_raises(bad_nc):
    mod = _import_module()
    with pytest.raises(ConfigurationError) as exc:
        mod.SSDMobileNetV3({"input_size": 320, "num_classes": bad_nc})
    assert any("num_classes" in v for v in exc.value.violations)


@requires_torchvision
def test_iou_threshold_out_of_range_raises():
    mod = _import_module()
    with pytest.raises(ConfigurationError):
        mod.SSDMobileNetV3({
            "input_size": 320, "num_classes": 5, "iou_threshold": 1.5,
        })


@requires_torchvision
def test_paired_discriminative_lr_required():
    """backbone_lr without head_lr (or vice versa) must error."""
    mod = _import_module()
    with pytest.raises(ConfigurationError):
        mod.SSDMobileNetV3({
            "input_size": 320, "num_classes": 5, "backbone_lr": 1e-5,
        })
    with pytest.raises(ConfigurationError):
        mod.SSDMobileNetV3({
            "input_size": 320, "num_classes": 5, "head_lr": 1e-3,
        })


# ---------------------------------------------------------------------------
# Build + forward / train_step contract (torchvision required)
# ---------------------------------------------------------------------------


@requires_torchvision
@pytest.mark.parametrize("input_size", [320, 640])
def test_build_succeeds_for_both_sizes(input_size):
    """Constructing the wrapper builds a torchvision SSD with the requested size."""
    mod = _import_module()
    detector = mod.SSDMobileNetV3({
        "input_size": input_size,
        "num_classes": 5,
        "pretrained_backbone": False,  # avoid network download in tests
    })
    from torchvision.models.detection.ssd import SSD
    assert isinstance(detector._model, SSD)
    assert detector.input_size == input_size


@requires_torchvision
def test_classification_head_output_matches_num_classes_plus_background():
    """Final cls conv must output ``num_anchors_per_location * (num_classes+1)`` channels.

    torchvision's ``SSDLiteClassificationHead`` doesn't expose ``num_classes``
    as an attribute; we introspect the final 1x1 Conv's out_channels of every
    per-level head and divide by the per-level num_anchors.
    """
    import torch.nn as nn
    mod = _import_module()
    num_classes = 5
    detector = mod.SSDMobileNetV3({
        "input_size": 320,
        "num_classes": num_classes,
        "pretrained_backbone": False,
    })
    head = detector._model.head.classification_head
    anchors_per_level = (
        detector._model.anchor_generator.num_anchors_per_location()
    )
    for level_idx, per_level_head in enumerate(head.module_list):
        # Each per-level head is a Sequential ending in a 1x1 Conv2d.
        final_conv = None
        for m in reversed(list(per_level_head.modules())):
            if isinstance(m, nn.Conv2d):
                final_conv = m
                break
        assert final_conv is not None
        expected = anchors_per_level[level_idx] * (num_classes + 1)
        assert final_conv.out_channels == expected, (
            f"level {level_idx}: expected {expected} out_channels, "
            f"got {final_conv.out_channels}"
        )


@requires_torchvision
def test_forward_returns_pixel_xyxy_boxes():
    """forward() must return boxes in pixel space, not normalized."""
    import torch
    mod = _import_module()
    detector = mod.SSDMobileNetV3({
        "input_size": 320,
        "num_classes": 5,
        "pretrained_backbone": False,
        "score_threshold": 0.0,  # force at least some outputs
        "confidence_threshold": 0.0,
    })
    detector.to_device(torch.device("cpu"))
    batch = torch.rand(1, 3, 320, 320)
    outputs = detector.forward(batch)
    assert isinstance(outputs, list) and len(outputs) == 1
    pred = outputs[0]
    assert set(pred.keys()) == {"boxes", "labels", "scores"}
    if pred["boxes"].numel() > 0:
        # Pixel coords must be in (0, 320), not (0, 1).
        assert float(pred["boxes"].max()) > 1.5, (
            "boxes appear normalized; expected pixel xyxy in input-size space"
        )
        assert float(pred["boxes"].max()) <= 320.0 + 1e-3


@requires_torchvision
def test_forward_labels_are_zero_indexed():
    """forward() should shift labels back to 0..num_classes-1."""
    import torch
    mod = _import_module()
    detector = mod.SSDMobileNetV3({
        "input_size": 320,
        "num_classes": 5,
        "pretrained_backbone": False,
        "score_threshold": 0.0,
        "confidence_threshold": 0.0,
    })
    detector.to_device(torch.device("cpu"))
    batch = torch.rand(1, 3, 320, 320)
    outputs = detector.forward(batch)
    labels = outputs[0]["labels"]
    if labels.numel() > 0:
        assert int(labels.min()) >= 0
        assert int(labels.max()) <= 4


@requires_torchvision
def test_train_step_shifts_labels_for_background():
    """train_step must add +1 to labels before calling the SSD."""
    import torch

    mod = _import_module()
    detector = mod.SSDMobileNetV3({
        "input_size": 320,
        "num_classes": 5,
        "pretrained_backbone": False,
    })

    captured = []
    original_forward = detector._model.forward

    def patched_forward(images, targets=None):
        if targets is not None:
            captured.append([
                {"labels": t["labels"].clone()} for t in targets
            ])
            loss = torch.tensor(0.5, requires_grad=True)
            return {"classification": loss, "bbox_regression": loss}
        return original_forward(images)

    detector._model.forward = patched_forward
    # Batch size >=2 so BN is happy even in eval-free training path.
    images = [torch.rand(3, 320, 320), torch.rand(3, 320, 320)]
    targets = [
        {"boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]]),
         "labels": torch.tensor([0])},
        {"boxes": torch.tensor([[50.0, 50.0, 200.0, 200.0]]),
         "labels": torch.tensor([4])},
    ]
    out = detector.train_step(images, targets)
    assert "loss_tensor" in out
    assert captured, "patched forward was not invoked"
    # Labels must have been shifted by +1.
    assert int(captured[0][0]["labels"][0]) == 1
    assert int(captured[0][1]["labels"][0]) == 5


@requires_torchvision
def test_train_step_handles_empty_batch():
    """Empty image list should yield a finite zero loss with grad_fn."""
    import torch
    mod = _import_module()
    detector = mod.SSDMobileNetV3({
        "input_size": 320, "num_classes": 5, "pretrained_backbone": False,
    })
    out = detector.train_step([], [])
    assert "loss_tensor" in out
    assert float(out["loss_tensor"]) == 0.0


@requires_torchvision
def test_train_step_filters_degenerate_boxes():
    """Boxes with non-positive w or h must be filtered before torchvision call."""
    import torch
    mod = _import_module()
    detector = mod.SSDMobileNetV3({
        "input_size": 320, "num_classes": 5, "pretrained_backbone": False,
    })

    captured = []
    def patched_forward(images, targets=None):
        if targets is not None:
            captured.append([t["boxes"].clone() for t in targets])
            loss = torch.tensor(0.5, requires_grad=True)
            return {"classification": loss, "bbox_regression": loss}
        return detector._model.forward(images)

    detector._model.forward = patched_forward

    # Use batch=2 so BatchNorm doesn't error in training mode.
    images = [torch.rand(3, 320, 320), torch.rand(3, 320, 320)]
    targets = [
        {
            "boxes": torch.tensor([
                [10.0, 10.0, 100.0, 100.0],  # valid
                [50.0, 50.0, 50.0, 50.0],    # zero w/h -> drop
                [20.0, 20.0, 10.0, 10.0],    # negative w/h -> drop
            ]),
            "labels": torch.tensor([0, 1, 2]),
        },
        {
            "boxes": torch.tensor([[5.0, 5.0, 50.0, 50.0]]),
            "labels": torch.tensor([0]),
        },
    ]
    detector.train_step(images, targets)
    assert captured
    assert captured[0][0].shape == (1, 4), "degenerate boxes were not filtered"
    assert captured[0][1].shape == (1, 4)


# ---------------------------------------------------------------------------
# Discriminative LR & freeze schedule
# ---------------------------------------------------------------------------


@requires_torchvision
def test_get_parameters_flat_when_no_discriminative_lr():
    mod = _import_module()
    import torch
    detector = mod.SSDMobileNetV3({
        "input_size": 320, "num_classes": 5, "pretrained_backbone": False,
    })
    params = detector.get_parameters()
    assert isinstance(params, list)
    assert all(isinstance(p, torch.nn.Parameter) for p in params)


@requires_torchvision
def test_get_parameters_returns_groups_for_discriminative_lr():
    mod = _import_module()
    detector = mod.SSDMobileNetV3({
        "input_size": 320,
        "num_classes": 5,
        "pretrained_backbone": False,
        "backbone_lr": 1e-5,
        "head_lr": 1e-3,
    })
    groups = detector.get_parameters()
    assert isinstance(groups, list)
    assert all(isinstance(g, dict) for g in groups)
    lrs = sorted(g["lr"] for g in groups if g["params"])
    assert lrs == [1e-5, 1e-3]


@requires_torchvision
def test_freeze_backbone_epochs_initial_freeze():
    """When freeze_backbone_epochs > 0, backbone params start frozen."""
    mod = _import_module()
    detector = mod.SSDMobileNetV3({
        "input_size": 320,
        "num_classes": 5,
        "pretrained_backbone": False,
        "freeze_backbone_epochs": 3,
    })
    backbone_trainable = [
        p.requires_grad for p in detector._model.backbone.parameters()
    ]
    assert not any(backbone_trainable), "backbone should start frozen"
    head_trainable = [
        p.requires_grad for p in detector._model.head.parameters()
    ]
    assert any(head_trainable), "head should remain trainable"


@requires_torchvision
def test_on_epoch_start_unfreezes_and_flags_rebuild():
    mod = _import_module()
    detector = mod.SSDMobileNetV3({
        "input_size": 320,
        "num_classes": 5,
        "pretrained_backbone": False,
        "freeze_backbone_epochs": 3,
    })
    assert detector.requires_optimizer_rebuild() is False
    # During the freeze window, calling on_epoch_start is a no-op transition.
    detector.on_epoch_start(0)
    assert detector.requires_optimizer_rebuild() is False
    # At epoch 3, backbone unfreezes and the flag goes high.
    detector.on_epoch_start(3)
    assert detector.requires_optimizer_rebuild() is True
    detector.acknowledge_optimizer_rebuild()
    assert detector.requires_optimizer_rebuild() is False
    backbone_trainable = [
        p.requires_grad for p in detector._model.backbone.parameters()
    ]
    assert any(backbone_trainable), "backbone should be unfrozen after epoch 3"


@requires_torchvision
def test_on_epoch_start_no_op_when_freeze_disabled():
    mod = _import_module()
    detector = mod.SSDMobileNetV3({
        "input_size": 320, "num_classes": 5, "pretrained_backbone": False,
    })
    detector.on_epoch_start(0)
    detector.on_epoch_start(10)
    assert detector.requires_optimizer_rebuild() is False


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


@requires_torchvision
def test_save_and_load_checkpoint_roundtrip(tmp_path: Path):
    import torch
    mod = _import_module()
    config = {
        "input_size": 320, "num_classes": 5, "pretrained_backbone": False,
    }
    src = mod.SSDMobileNetV3(config)
    src_state = {k: v.clone() for k, v in src._model.state_dict().items()}

    ckpt = tmp_path / "ckpt.pt"
    src.save_checkpoint(ckpt, epoch=3, metrics={"map_50": 0.42})
    assert ckpt.exists()

    dst = mod.SSDMobileNetV3(config)
    dst.load_checkpoint(ckpt)
    dst_state = dst._model.state_dict()

    # State dict contents must round-trip exactly.
    assert set(src_state.keys()) == set(dst_state.keys())
    for k in src_state:
        assert torch.equal(src_state[k], dst_state[k]), f"mismatch in {k}"


def test_load_nonexistent_checkpoint_raises(tmp_path: Path):
    """Even without torch we can verify the file-not-found path."""
    if not HAS_TORCHVISION:
        pytest.skip("torchvision required for instantiation")
    mod = _import_module()
    detector = mod.SSDMobileNetV3({
        "input_size": 320, "num_classes": 5, "pretrained_backbone": False,
    })
    with pytest.raises(FileNotFoundError):
        detector.load_checkpoint(tmp_path / "does_not_exist.pt")
