"""Property-based tests for BaseDetector default mode/device methods.

Feature: generic-evaluation-script
Property 19: BaseDetector default mode/device methods delegate correctly

For any detector object, the default ``set_eval_mode``, ``set_train_mode``, and
``to_device`` implementations operate on the ``_model`` attribute when present
and otherwise on the ``model`` attribute (with ``_model`` taking precedence when
both are present), and are no-ops when neither attribute is present.

These tests use a conforming fake ``BaseDetector`` whose ``_model`` / ``model``
attributes are lightweight fakes exposing ``train`` / ``eval`` / ``to`` so the
tests run without GPUs or real checkpoints.

**Validates: Requirements 2.2**
"""

from pathlib import Path
from typing import List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from model.models.registry import BaseDetector


# ---------------------------------------------------------------------------
# Conforming fakes
# ---------------------------------------------------------------------------


class FakeModule:
    """Minimal nn.Module-like stand-in that records train/eval/to calls."""

    def __init__(self) -> None:
        self.train_calls = 0
        self.eval_calls = 0
        self.to_calls: List[object] = []

    def train(self, mode: bool = True):
        self.train_calls += 1
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def to(self, device):
        self.to_calls.append(device)
        return self


class FakeDetector(BaseDetector):
    """Concrete BaseDetector that implements only the abstract methods.

    It deliberately does NOT override ``set_train_mode`` / ``set_eval_mode`` /
    ``to_device`` so that the inherited default implementations are exercised.
    """

    def forward(self, images):  # pragma: no cover - not used by these tests
        return []

    def get_config_schema(self) -> dict:
        return {}

    def load_checkpoint(self, path: Path) -> None:  # pragma: no cover
        pass

    def save_checkpoint(self, path: Path) -> None:  # pragma: no cover
        pass


# Devices the script might request. Plain strings keep the test GPU-free while
# still letting us assert the exact value forwarded to ``model.to(...)``.
DEVICES = st.sampled_from(["cpu", "cuda", "cuda:0", "cuda:1", "meta", "mps"])


# ---------------------------------------------------------------------------
# Property 19
# ---------------------------------------------------------------------------


class TestProperty19BaseDetectorDefaults:
    """Property 19: BaseDetector default mode/device methods delegate correctly.

    **Validates: Requirements 2.2**
    """

    @given(device=DEVICES)
    @settings(max_examples=100)
    def test_underscore_model_takes_precedence_when_both_present(self, device):
        # Feature: generic-evaluation-script, Property 19: BaseDetector default mode/device methods delegate correctly
        """When both `_model` and `model` are present, defaults operate on `_model`."""
        det = FakeDetector()
        underscore = FakeModule()
        public = FakeModule()
        det._model = underscore
        det.model = public

        det.set_train_mode()
        det.set_eval_mode()
        det.to_device(device)

        # `_model` is the one operated on.
        assert underscore.train_calls == 1
        assert underscore.eval_calls == 1
        assert underscore.to_calls == [device]

        # `model` is left untouched because `_model` takes precedence.
        assert public.train_calls == 0
        assert public.eval_calls == 0
        assert public.to_calls == []

    @given(device=DEVICES)
    @settings(max_examples=100)
    def test_operates_on_underscore_model_when_only_underscore_present(self, device):
        # Feature: generic-evaluation-script, Property 19: BaseDetector default mode/device methods delegate correctly
        """When only `_model` is present, defaults operate on `_model`."""
        det = FakeDetector()
        underscore = FakeModule()
        det._model = underscore

        det.set_train_mode()
        det.set_eval_mode()
        det.to_device(device)

        assert underscore.train_calls == 1
        assert underscore.eval_calls == 1
        assert underscore.to_calls == [device]

    @given(device=DEVICES)
    @settings(max_examples=100)
    def test_operates_on_public_model_when_only_public_present(self, device):
        # Feature: generic-evaluation-script, Property 19: BaseDetector default mode/device methods delegate correctly
        """When only `model` is present (no `_model`), defaults operate on `model`."""
        det = FakeDetector()
        public = FakeModule()
        det.model = public

        det.set_train_mode()
        det.set_eval_mode()
        det.to_device(device)

        assert public.train_calls == 1
        assert public.eval_calls == 1
        assert public.to_calls == [device]

    @given(device=DEVICES)
    @settings(max_examples=100)
    def test_falls_back_to_public_model_when_underscore_is_none(self, device):
        # Feature: generic-evaluation-script, Property 19: BaseDetector default mode/device methods delegate correctly
        """A None `_model` is treated as absent, so defaults fall back to `model`."""
        det = FakeDetector()
        public = FakeModule()
        det._model = None
        det.model = public

        det.set_train_mode()
        det.set_eval_mode()
        det.to_device(device)

        assert public.train_calls == 1
        assert public.eval_calls == 1
        assert public.to_calls == [device]

    @given(device=DEVICES)
    @settings(max_examples=100)
    def test_noop_when_neither_attribute_present(self, device):
        # Feature: generic-evaluation-script, Property 19: BaseDetector default mode/device methods delegate correctly
        """When neither `_model` nor `model` is present, defaults are no-ops."""
        det = FakeDetector()

        assert det._underlying_model() is None

        # Must not raise even though there is no underlying model.
        det.set_train_mode()
        det.set_eval_mode()
        det.to_device(device)

    @given(device=DEVICES)
    @settings(max_examples=100)
    def test_to_device_updates_device_attribute_when_present(self, device):
        # Feature: generic-evaluation-script, Property 19: BaseDetector default mode/device methods delegate correctly
        """`to_device` updates `self._device` when that attribute exists."""
        det = FakeDetector()
        det._model = FakeModule()
        det._device = "cpu"

        det.to_device(device)

        assert det._device == device

    @given(device=DEVICES)
    @settings(max_examples=100)
    def test_to_device_updates_device_attribute_without_underlying_model(self, device):
        # Feature: generic-evaluation-script, Property 19: BaseDetector default mode/device methods delegate correctly
        """`to_device` updates `self._device` even when no model is present."""
        det = FakeDetector()
        det._device = "cpu"

        det.to_device(device)

        assert det._device == device
