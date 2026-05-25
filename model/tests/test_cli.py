"""Unit tests for CLI commands and argument parsing."""

import logging
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from model.cli import create_parser, handle_list_models, main


class TestArgumentParsing:
    """Test argument parsing for all subcommands."""

    def test_train_command_requires_config(self):
        """Train command requires --config argument."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["train"])

    def test_train_command_parses_config(self):
        """Train command correctly parses --config path."""
        parser = create_parser()
        args = parser.parse_args(["train", "--config", "path/to/config.yaml"])
        assert args.command == "train"
        assert args.config == "path/to/config.yaml"

    def test_evaluate_command_requires_checkpoint_and_dataset(self):
        """Evaluate command requires both --checkpoint and --dataset."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["evaluate"])
        with pytest.raises(SystemExit):
            parser.parse_args(["evaluate", "--checkpoint", "model.pt"])
        with pytest.raises(SystemExit):
            parser.parse_args(["evaluate", "--dataset", "/data"])

    def test_evaluate_command_parses_args(self):
        """Evaluate command correctly parses checkpoint and dataset paths."""
        parser = create_parser()
        args = parser.parse_args([
            "evaluate", "--checkpoint", "model.pt", "--dataset", "/data/test"
        ])
        assert args.command == "evaluate"
        assert args.checkpoint == "model.pt"
        assert args.dataset == "/data/test"

    def test_predict_command_requires_all_args(self):
        """Predict command requires --checkpoint, --input, and --output."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["predict"])
        with pytest.raises(SystemExit):
            parser.parse_args(["predict", "--checkpoint", "model.pt"])
        with pytest.raises(SystemExit):
            parser.parse_args([
                "predict", "--checkpoint", "model.pt", "--input", "/images"
            ])

    def test_predict_command_parses_args(self):
        """Predict command correctly parses all arguments."""
        parser = create_parser()
        args = parser.parse_args([
            "predict",
            "--checkpoint", "model.pt",
            "--input", "/images/test",
            "--output", "/output/results",
        ])
        assert args.command == "predict"
        assert args.checkpoint == "model.pt"
        assert args.input == "/images/test"
        assert args.output == "/output/results"

    def test_list_models_command(self):
        """List-models command requires no additional arguments."""
        parser = create_parser()
        args = parser.parse_args(["list-models"])
        assert args.command == "list-models"

    def test_verbose_flag_default_false(self):
        """Verbose flag defaults to False."""
        parser = create_parser()
        args = parser.parse_args(["list-models"])
        assert args.verbose is False

    def test_verbose_flag_long(self):
        """--verbose flag is parsed correctly."""
        parser = create_parser()
        args = parser.parse_args(["--verbose", "list-models"])
        assert args.verbose is True

    def test_verbose_flag_short(self):
        """-v flag is parsed correctly."""
        parser = create_parser()
        args = parser.parse_args(["-v", "list-models"])
        assert args.verbose is True

    def test_no_command_returns_none(self):
        """No command sets command to None."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.command is None


class TestVerboseLogging:
    """Test that --verbose flag sets DEBUG logging level."""

    def test_verbose_sets_debug_level(self):
        """--verbose flag configures logging to DEBUG level."""
        with patch("model.cli.handle_list_models") as mock_handler:
            # Reset logging to allow reconfiguration
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

            main(["--verbose", "list-models"])

            # Check that the root logger level is DEBUG
            assert root_logger.level == logging.DEBUG
            mock_handler.assert_called_once()

    def test_default_sets_info_level(self):
        """Without --verbose, logging is set to INFO level."""
        with patch("model.cli.handle_list_models") as mock_handler:
            # Reset logging to allow reconfiguration
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

            main(["list-models"])

            assert root_logger.level == logging.INFO
            mock_handler.assert_called_once()


class TestErrorHandling:
    """Test error handling with non-zero exit and stderr output."""

    def test_handler_exception_exits_nonzero(self):
        """When a handler raises an exception, CLI exits with code 1."""
        with patch("model.cli.handle_list_models", side_effect=RuntimeError("test error")):
            with pytest.raises(SystemExit) as exc_info:
                main(["list-models"])
            assert exc_info.value.code == 1

    def test_handler_exception_prints_to_stderr(self, capsys):
        """When a handler raises an exception, error message goes to stderr."""
        with patch("model.cli.handle_list_models", side_effect=RuntimeError("something broke")):
            with pytest.raises(SystemExit):
                main(["list-models"])
            captured = capsys.readouterr()
            assert "something broke" in captured.err

    def test_no_command_exits_zero(self):
        """No command prints help and exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0


class TestListModelsCommand:
    """Test the list-models command output."""

    def test_list_models_prints_registered_models(self, capsys):
        """list-models prints registered model identifiers."""
        with patch("model.models.registry.ModelRegistry.list_models",
                   return_value=["ssd_mobilenetv3", "yolov6"]):
            with patch.dict("sys.modules", {
                "model.models.ssd_mobilenet": MagicMock(),
                "model.models.yolov6_wrapper": MagicMock(),
            }):
                main(["list-models"])

        captured = capsys.readouterr()
        assert "ssd_mobilenetv3" in captured.out
        assert "yolov6" in captured.out


class TestTrainCommand:
    """Test the train command wiring."""

    def test_train_missing_config_file_exits_nonzero(self, tmp_path):
        """Train with non-existent config file exits with error."""
        fake_config = tmp_path / "nonexistent.yaml"
        with pytest.raises(SystemExit) as exc_info:
            main(["train", "--config", str(fake_config)])
        assert exc_info.value.code == 1

    def test_train_missing_model_type_exits_nonzero(self, tmp_path):
        """Train with config missing model.type exits with error."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("training:\n  epochs: 10\n")
        with pytest.raises(SystemExit) as exc_info:
            main(["train", "--config", str(config_file)])
        assert exc_info.value.code == 1


class TestEvaluateCommand:
    """Test the evaluate command wiring."""

    def test_evaluate_missing_checkpoint_exits_nonzero(self, tmp_path, capsys):
        """Evaluate with non-existent checkpoint exits with error."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        with pytest.raises(SystemExit) as exc_info:
            main([
                "evaluate",
                "--checkpoint", str(tmp_path / "nonexistent.pt"),
                "--dataset", str(dataset_dir),
            ])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Checkpoint not found" in captured.err

    def test_evaluate_missing_dataset_exits_nonzero(self, tmp_path, capsys):
        """Evaluate with non-existent dataset path exits with error."""
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_text("fake")
        with pytest.raises(SystemExit) as exc_info:
            main([
                "evaluate",
                "--checkpoint", str(checkpoint),
                "--dataset", str(tmp_path / "nonexistent"),
            ])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Dataset path not found" in captured.err


class TestPredictCommand:
    """Test the predict command wiring."""

    def test_predict_missing_checkpoint_exits_nonzero(self, tmp_path, capsys):
        """Predict with non-existent checkpoint exits with error."""
        input_dir = tmp_path / "images"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        with pytest.raises(SystemExit) as exc_info:
            main([
                "predict",
                "--checkpoint", str(tmp_path / "nonexistent.pt"),
                "--input", str(input_dir),
                "--output", str(output_dir),
            ])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Checkpoint not found" in captured.err

    def test_predict_missing_input_exits_nonzero(self, tmp_path, capsys):
        """Predict with non-existent input path exits with error."""
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_text("fake")
        output_dir = tmp_path / "output"
        with pytest.raises(SystemExit) as exc_info:
            main([
                "predict",
                "--checkpoint", str(checkpoint),
                "--input", str(tmp_path / "nonexistent"),
                "--output", str(output_dir),
            ])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Input path not found" in captured.err
