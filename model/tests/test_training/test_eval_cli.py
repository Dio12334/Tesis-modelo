"""Unit tests for CLI argument parsing and deprecation warnings.

Feature: generic-evaluation-script, Task 13.2

These tests verify that:
- All prior CLI arguments are accepted and parsed correctly (Requirement 17.1).
- Each argument maps to the correct override path in the structured config
  sections (model, dataset, evaluation, checkpoint).
- A deprecated argument (--model-type when --config is supplied) emits a
  WARNING and continues execution (Requirement 17.2).
"""

import logging
from unittest.mock import patch

import pytest

from model.training.evaluate_detection import build_arg_parser, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(args: list[str]):
    """Parse a list of CLI argument strings using build_arg_parser."""
    parser = build_arg_parser()
    return parser.parse_args(args)


# ---------------------------------------------------------------------------
# Tests: build_arg_parser produces correct namespace values
# ---------------------------------------------------------------------------


class TestBuildArgParserBasicArguments:
    """All prior arguments are accepted and produce the expected namespace values.

    **Validates: Requirements 17.1**
    """

    def test_config_argument(self):
        args = _parse(["--config", "path/to/config.yaml"])
        assert args.config == "path/to/config.yaml"

    def test_checkpoint_argument(self):
        args = _parse(["--checkpoint", "/models/best.pt"])
        assert args.checkpoint == "/models/best.pt"

    def test_run_id_argument(self):
        args = _parse(["--run-id", "abc-123-uuid"])
        assert args.run_id == "abc-123-uuid"

    def test_checkpoint_dir_argument(self):
        args = _parse(["--checkpoint-dir", "/custom/checkpoints"])
        assert args.checkpoint_dir == "/custom/checkpoints"

    def test_checkpoint_dir_default(self):
        args = _parse([])
        assert args.checkpoint_dir == "./checkpoints"

    def test_model_type_argument(self):
        args = _parse(["--model-type", "yolo26"])
        assert args.model_type == "yolo26"

    def test_input_size_argument(self):
        args = _parse(["--input-size", "640"])
        assert args.input_size == 640

    def test_num_classes_argument(self):
        args = _parse(["--num-classes", "4"])
        assert args.num_classes == 4

    def test_dataset_argument(self):
        args = _parse(["--dataset", "/data/rdd2022"])
        assert args.dataset == "/data/rdd2022"

    def test_split_argument_train(self):
        args = _parse(["--split", "train"])
        assert args.split == "train"

    def test_split_argument_val(self):
        args = _parse(["--split", "val"])
        assert args.split == "val"

    def test_split_argument_test(self):
        args = _parse(["--split", "test"])
        assert args.split == "test"

    def test_val_split_argument(self):
        args = _parse(["--val-split", "0.15"])
        assert args.val_split == pytest.approx(0.15)

    def test_confidence_argument(self):
        args = _parse(["--confidence", "0.3"])
        assert args.confidence == pytest.approx(0.3)

    def test_iou_argument(self):
        args = _parse(["--iou", "0.45"])
        assert args.iou == pytest.approx(0.45)

    def test_output_dir_argument(self):
        args = _parse(["--output-dir", "/results/eval"])
        assert args.output_dir == "/results/eval"

    def test_verbose_long_flag(self):
        args = _parse(["--verbose"])
        assert args.verbose is True

    def test_verbose_short_flag(self):
        args = _parse(["-v"])
        assert args.verbose is True

    def test_verbose_default_false(self):
        args = _parse([])
        assert args.verbose is False

    def test_split_rejects_invalid_choice(self):
        """--split only accepts train, val, or test."""
        with pytest.raises(SystemExit):
            _parse(["--split", "invalid"])


# ---------------------------------------------------------------------------
# Tests: main() maps arguments to correct override paths
# ---------------------------------------------------------------------------


class TestMainOverrideMapping:
    """main() maps CLI arguments into the correct structured override paths.

    **Validates: Requirements 17.1**
    """

    @patch("model.training.evaluate_detection.evaluate")
    def test_checkpoint_maps_to_checkpoint_path(self, mock_evaluate):
        with patch(
            "sys.argv",
            ["evaluate_detection.py", "--checkpoint", "/models/best.pt"],
        ):
            main()
        _, kwargs = mock_evaluate.call_args
        overrides = kwargs["overrides"]
        assert overrides["checkpoint"]["path"] == "/models/best.pt"

    @patch("model.training.evaluate_detection.evaluate")
    def test_run_id_maps_to_checkpoint_run_id(self, mock_evaluate):
        with patch("sys.argv", ["evaluate_detection.py", "--run-id", "run-42"]):
            main()
        _, kwargs = mock_evaluate.call_args
        overrides = kwargs["overrides"]
        assert overrides["checkpoint"]["run_id"] == "run-42"

    @patch("model.training.evaluate_detection.evaluate")
    def test_checkpoint_dir_maps_to_checkpoint_checkpoint_dir(self, mock_evaluate):
        with patch(
            "sys.argv",
            ["evaluate_detection.py", "--checkpoint-dir", "/custom/dir"],
        ):
            main()
        _, kwargs = mock_evaluate.call_args
        overrides = kwargs["overrides"]
        assert overrides["checkpoint"]["checkpoint_dir"] == "/custom/dir"

    @patch("model.training.evaluate_detection.evaluate")
    def test_checkpoint_dir_default_not_in_overrides(self, mock_evaluate):
        """Default checkpoint-dir value is NOT placed in overrides."""
        with patch("sys.argv", ["evaluate_detection.py"]):
            main()
        _, kwargs = mock_evaluate.call_args
        overrides = kwargs["overrides"]
        # When checkpoint-dir is the default, it should not appear in overrides
        assert "checkpoint" not in overrides or "checkpoint_dir" not in overrides.get(
            "checkpoint", {}
        )

    @patch("model.training.evaluate_detection.evaluate")
    def test_model_type_maps_to_model_type(self, mock_evaluate):
        with patch(
            "sys.argv", ["evaluate_detection.py", "--model-type", "ssd_mobilenet"]
        ):
            main()
        _, kwargs = mock_evaluate.call_args
        overrides = kwargs["overrides"]
        assert overrides["model"]["type"] == "ssd_mobilenet"

    @patch("model.training.evaluate_detection.evaluate")
    def test_input_size_maps_to_both_paths(self, mock_evaluate):
        """--input-size maps to both evaluation.input_size and model.config.input_size."""
        with patch("sys.argv", ["evaluate_detection.py", "--input-size", "320"]):
            main()
        _, kwargs = mock_evaluate.call_args
        overrides = kwargs["overrides"]
        assert overrides["model"]["config"]["input_size"] == 320
        assert overrides["evaluation"]["input_size"] == 320

    @patch("model.training.evaluate_detection.evaluate")
    def test_num_classes_maps_to_model_config_num_classes(self, mock_evaluate):
        with patch("sys.argv", ["evaluate_detection.py", "--num-classes", "7"]):
            main()
        _, kwargs = mock_evaluate.call_args
        overrides = kwargs["overrides"]
        assert overrides["model"]["config"]["num_classes"] == 7

    @patch("model.training.evaluate_detection.evaluate")
    def test_dataset_maps_to_dataset_path(self, mock_evaluate):
        with patch(
            "sys.argv", ["evaluate_detection.py", "--dataset", "/data/rdd2022"]
        ):
            main()
        _, kwargs = mock_evaluate.call_args
        overrides = kwargs["overrides"]
        assert overrides["dataset"]["path"] == "/data/rdd2022"

    @patch("model.training.evaluate_detection.evaluate")
    def test_split_maps_to_evaluation_split(self, mock_evaluate):
        with patch("sys.argv", ["evaluate_detection.py", "--split", "val"]):
            main()
        _, kwargs = mock_evaluate.call_args
        overrides = kwargs["overrides"]
        assert overrides["evaluation"]["split"] == "val"

    @patch("model.training.evaluate_detection.evaluate")
    def test_val_split_maps_to_evaluation_val_split(self, mock_evaluate):
        with patch("sys.argv", ["evaluate_detection.py", "--val-split", "0.2"]):
            main()
        _, kwargs = mock_evaluate.call_args
        overrides = kwargs["overrides"]
        assert overrides["evaluation"]["val_split"] == pytest.approx(0.2)

    @patch("model.training.evaluate_detection.evaluate")
    def test_confidence_maps_to_evaluation_confidence_threshold(self, mock_evaluate):
        with patch("sys.argv", ["evaluate_detection.py", "--confidence", "0.5"]):
            main()
        _, kwargs = mock_evaluate.call_args
        overrides = kwargs["overrides"]
        assert overrides["evaluation"]["confidence_threshold"] == pytest.approx(0.5)

    @patch("model.training.evaluate_detection.evaluate")
    def test_iou_maps_to_evaluation_iou_threshold(self, mock_evaluate):
        with patch("sys.argv", ["evaluate_detection.py", "--iou", "0.6"]):
            main()
        _, kwargs = mock_evaluate.call_args
        overrides = kwargs["overrides"]
        assert overrides["evaluation"]["iou_threshold"] == pytest.approx(0.6)

    @patch("model.training.evaluate_detection.evaluate")
    def test_output_dir_maps_to_evaluation_output_dir(self, mock_evaluate):
        with patch(
            "sys.argv", ["evaluate_detection.py", "--output-dir", "/results"]
        ):
            main()
        _, kwargs = mock_evaluate.call_args
        overrides = kwargs["overrides"]
        assert overrides["evaluation"]["output_dir"] == "/results"

    @patch("model.training.evaluate_detection.evaluate")
    def test_config_passed_as_config_path(self, mock_evaluate):
        """--config is passed as config_path kwarg, not as an override."""
        with patch(
            "sys.argv", ["evaluate_detection.py", "--config", "my_config.yaml"]
        ):
            main()
        _, kwargs = mock_evaluate.call_args
        assert kwargs["config_path"] == "my_config.yaml"

    @patch("model.training.evaluate_detection.evaluate")
    def test_no_args_produces_empty_overrides(self, mock_evaluate):
        """When no CLI arguments are given, overrides dict is empty."""
        with patch("sys.argv", ["evaluate_detection.py"]):
            main()
        _, kwargs = mock_evaluate.call_args
        overrides = kwargs["overrides"]
        assert overrides == {}


# ---------------------------------------------------------------------------
# Tests: Deprecation warning for --model-type when --config is supplied
# ---------------------------------------------------------------------------


class TestDeprecationWarning:
    """A deprecated argument emits a WARNING and continues execution.

    **Validates: Requirements 17.2**
    """

    @patch("model.training.evaluate_detection.evaluate")
    def test_model_type_with_config_emits_warning(self, mock_evaluate, caplog):
        """--model-type with --config logs a deprecation WARNING."""
        with patch(
            "sys.argv",
            [
                "evaluate_detection.py",
                "--config",
                "config.yaml",
                "--model-type",
                "yolo26",
            ],
        ):
            with caplog.at_level(logging.WARNING):
                main()

        # A WARNING about --model-type deprecation was emitted.
        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any("--model-type" in msg and "deprecated" in msg.lower() for msg in warning_messages)

    @patch("model.training.evaluate_detection.evaluate")
    def test_model_type_with_config_continues_execution(self, mock_evaluate, caplog):
        """Deprecated argument does not prevent evaluate() from being called."""
        with patch(
            "sys.argv",
            [
                "evaluate_detection.py",
                "--config",
                "config.yaml",
                "--model-type",
                "yolo26",
            ],
        ):
            with caplog.at_level(logging.WARNING):
                main()

        # evaluate() was still called despite the deprecation warning.
        mock_evaluate.assert_called_once()

    @patch("model.training.evaluate_detection.evaluate")
    def test_model_type_with_config_still_overrides(self, mock_evaluate, caplog):
        """The deprecated --model-type value is still used as an override."""
        with patch(
            "sys.argv",
            [
                "evaluate_detection.py",
                "--config",
                "config.yaml",
                "--model-type",
                "yolo26",
            ],
        ):
            with caplog.at_level(logging.WARNING):
                main()

        _, kwargs = mock_evaluate.call_args
        overrides = kwargs["overrides"]
        assert overrides["model"]["type"] == "yolo26"

    @patch("model.training.evaluate_detection.evaluate")
    def test_model_type_without_config_no_warning(self, mock_evaluate, caplog):
        """--model-type without --config does NOT emit a deprecation warning."""
        with patch(
            "sys.argv",
            ["evaluate_detection.py", "--model-type", "yolo26"],
        ):
            with caplog.at_level(logging.WARNING):
                main()

        # No deprecation warning should be emitted.
        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert not any(
            "--model-type" in msg and "deprecated" in msg.lower()
            for msg in warning_messages
        )

    @patch("model.training.evaluate_detection.evaluate")
    def test_deprecation_warning_mentions_replacement(self, mock_evaluate, caplog):
        """The deprecation warning mentions the replacement (config file)."""
        with patch(
            "sys.argv",
            [
                "evaluate_detection.py",
                "--config",
                "config.yaml",
                "--model-type",
                "yolo26",
            ],
        ):
            with caplog.at_level(logging.WARNING):
                main()

        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        deprecation_msgs = [
            msg for msg in warning_messages
            if "--model-type" in msg and "deprecated" in msg.lower()
        ]
        assert len(deprecation_msgs) >= 1
        # The warning should mention the replacement approach.
        assert any("config" in msg.lower() for msg in deprecation_msgs)
