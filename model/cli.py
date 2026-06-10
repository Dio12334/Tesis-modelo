"""Command-line interface for the Road Damage Evaluation Framework.

Provides subcommands for training, evaluation, inference, and model listing.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Checkpoint filenames tried in order of preference for inference
_INFER_CHECKPOINT_PREFERENCE = ["best_model.pt", "final_model.pt", "recovery.pt"]
_DEFAULT_CHECKPOINT_BASE = Path("checkpoints")


def _infer_model_type_from_path(checkpoint_path: Path, checkpoint_base_dir: Path) -> Optional[str]:
    """Return the model-type directory name if *checkpoint_path* lives under
    ``<checkpoint_base_dir>/<model_type>/…``, otherwise return None."""
    try:
        rel = checkpoint_path.resolve().relative_to(checkpoint_base_dir.resolve())
        if rel.parts:
            return rel.parts[0]
    except ValueError:
        pass
    return None


def _find_by_run_id(run_id: str, checkpoint_base_dir: Path) -> Tuple[Path, str]:
    """Locate the best available checkpoint for *run_id*.

    Searches ``<checkpoint_base_dir>/<model_type>/<run_id>.json`` across all
    model-type subdirectories.  Returns ``(checkpoint_path, model_type)``.
    """
    if not checkpoint_base_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_base_dir}")

    for model_dir in sorted(checkpoint_base_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        json_path = model_dir / f"{run_id}.json"
        if not json_path.exists():
            continue
        with json_path.open() as fh:
            metadata = json.load(fh)
        model_type = metadata.get("model_name")
        if not model_type:
            raise ValueError(
                f"Run metadata at {json_path} is missing the 'model_name' field."
            )
        run_dir = model_dir / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        for filename in _INFER_CHECKPOINT_PREFERENCE:
            pt_path = run_dir / filename
            if pt_path.exists():
                return pt_path, model_type
        raise FileNotFoundError(
            f"No .pt checkpoint file found in {run_dir}. "
            f"Looked for: {_INFER_CHECKPOINT_PREFERENCE}"
        )

    raise FileNotFoundError(
        f"No run found with ID '{run_id}' under {checkpoint_base_dir}"
    )


def _find_last_checkpoint(
    model_type_filter: Optional[str], checkpoint_base_dir: Path
) -> Tuple[Path, str]:
    """Find the most recently completed run's checkpoint.

    Scans all ``<run_id>.json`` metadata files and sorts by ``end_time``
    (ISO-8601 strings sort lexicographically).  Falls back to file mtime.
    Returns ``(checkpoint_path, model_type)``.
    """
    if not checkpoint_base_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_base_dir}")

    candidates = []
    for model_dir in sorted(checkpoint_base_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_type_filter and model_dir.name != model_type_filter:
            continue
        for json_path in model_dir.glob("*.json"):
            try:
                with json_path.open() as fh:
                    metadata = json.load(fh)
                end_time = metadata.get("end_time", "")
                model_name = metadata.get("model_name", model_dir.name)
                run_id = metadata.get("run_id", json_path.stem)
                candidates.append(
                    (end_time, json_path.stat().st_mtime, run_id, model_name, model_dir)
                )
            except (json.JSONDecodeError, OSError):
                continue

    if not candidates:
        scope = f" for model type '{model_type_filter}'" if model_type_filter else ""
        raise FileNotFoundError(
            f"No completed checkpoints found{scope} under {checkpoint_base_dir}"
        )

    # Primary sort: end_time string (ISO-8601 lex order, None sorts last); secondary: file mtime
    candidates.sort(key=lambda x: (x[0] or "", x[1]), reverse=True)
    _, _, run_id, model_name, model_dir = candidates[0]

    run_dir = model_dir / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    for filename in _INFER_CHECKPOINT_PREFERENCE:
        pt_path = run_dir / filename
        if pt_path.exists():
            return pt_path, model_name

    raise FileNotFoundError(
        f"No .pt checkpoint file found in {run_dir}. "
        f"Looked for: {_INFER_CHECKPOINT_PREFERENCE}"
    )


def _resolve_checkpoint_and_model_type(
    args: argparse.Namespace,
    checkpoint_base_dir: Path = _DEFAULT_CHECKPOINT_BASE,
) -> Tuple[Path, str]:
    """Resolve checkpoint path and model type from parsed CLI args.

    Exactly one of ``--checkpoint``, ``--run-id``, or ``--last`` must be set.
    Returns ``(checkpoint_path, model_type)``.
    """
    has_checkpoint = bool(getattr(args, "checkpoint", None))
    has_run_id = bool(getattr(args, "run_id", None))
    has_last = bool(getattr(args, "last", False))
    model_type_arg: Optional[str] = getattr(args, "model_type", None)

    specified = sum([has_checkpoint, has_run_id, has_last])
    if specified == 0:
        raise ValueError(
            "Must specify one of --checkpoint <path>, --run-id <id>, or --last."
        )
    if specified > 1:
        raise ValueError(
            "Only one of --checkpoint, --run-id, or --last may be specified at a time."
        )

    if has_last:
        checkpoint_path, model_type = _find_last_checkpoint(model_type_arg, checkpoint_base_dir)
        logger.info(
            "Using last checkpoint: %s (model: %s)", checkpoint_path, model_type
        )
        return checkpoint_path, model_type

    if has_run_id:
        checkpoint_path, model_type = _find_by_run_id(args.run_id, checkpoint_base_dir)
        logger.info(
            "Using checkpoint for run %s: %s (model: %s)",
            args.run_id,
            checkpoint_path,
            model_type,
        )
        return checkpoint_path, model_type

    # --checkpoint explicit path
    checkpoint_path = Path(args.checkpoint)
    if model_type_arg:
        model_type = model_type_arg
    else:
        model_type = _infer_model_type_from_path(checkpoint_path, checkpoint_base_dir)
        if not model_type:
            raise ValueError(
                f"Cannot determine model type from path: {checkpoint_path}\n"
                "The path does not match the expected structure "
                "<checkpoint_base>/<model_type>/...  "
                "Use --model-type <name> to specify it explicitly."
            )
    return checkpoint_path, model_type


def _add_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    """Add mutually-exclusive checkpoint selection arguments to *parser*."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a .pt checkpoint file.",
    )
    group.add_argument(
        "--run-id",
        type=str,
        dest="run_id",
        default=None,
        metavar="UUID",
        help="Run ID; auto-detects model type and picks the best available .pt file.",
    )
    group.add_argument(
        "--last",
        action="store_true",
        default=False,
        help="Use the most recently completed checkpoint.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        dest="model_type",
        default=None,
        metavar="NAME",
        help=(
            "Model type name (e.g. rt_detr, yolo26).  "
            "Required with --checkpoint when the path is outside the standard "
            "checkpoint directory.  Optionally scopes --last to one model type."
        ),
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="model",
        description="Road Damage Evaluation Framework - Train, evaluate, and run inference with detection models.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG) logging.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # train subcommand
    train_parser = subparsers.add_parser(
        "train",
        help="Train a detection model using a configuration file.",
    )
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration YAML file.",
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from a checkpoint. Provide a path to a .pt file or a run ID.",
    )

    # evaluate subcommand
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained model on a dataset.",
    )
    _add_checkpoint_args(eval_parser)
    eval_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset directory.",
    )

    # predict subcommand
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run inference on images using a trained model.",
    )
    _add_checkpoint_args(predict_parser)
    predict_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image or directory of images.",
    )
    predict_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory for annotated images.",
    )

    # list-models subcommand
    subparsers.add_parser(
        "list-models",
        help="List all registered model identifiers.",
    )

    return parser


def handle_train(args: argparse.Namespace) -> None:
    """Handle the train subcommand.

    Delegates to the real PyTorch training script (train_detection.py)
    which performs actual GPU training with gradient updates, checkpoint
    saving, and experiment tracking.

    Args:
        args: Parsed command-line arguments.
    """
    from model.training.train_detection import train

    config_path = args.config
    verbose = getattr(args, "verbose", False)
    resume_from = getattr(args, "resume", None)

    logger.info("Starting real PyTorch training with config: %s", config_path)
    results = train(config_path=config_path, verbose=verbose, resume_from=resume_from)

    def _fmt(val):
        return f"{val:.4f}" if isinstance(val, (int, float)) else str(val)

    print(f"\nTraining complete.")
    print(f"  Final train loss: {_fmt(results.get('final_train_loss', 'N/A'))}")
    print(f"  Final val loss:   {_fmt(results.get('final_val_loss', 'N/A'))}")
    print(f"  Best val loss:    {_fmt(results.get('best_val_loss', 'N/A'))}")
    print(f"  Best epoch:       {results.get('best_epoch', 'N/A')}")
    print(f"  Total epochs:     {results.get('total_epochs', 'N/A')}")
    if "run_id" in results:
        print(f"  Run ID:           {results['run_id']}")


def handle_evaluate(args: argparse.Namespace) -> None:
    """Handle the evaluate subcommand.

    Loads a model checkpoint, runs evaluation on the dataset,
    saves the report, and logs to ExperimentTracker.

    Args:
        args: Parsed command-line arguments.
    """
    import model.models  # noqa: F401 — registers all model wrappers
    from model.datasets.rdd2022 import RDD2022Dataset
    from model.evaluation.engine import EvaluationEngine
    from model.models.registry import ModelRegistry
    from model.tracking.tracker import ExperimentTracker

    checkpoint_path, model_type = _resolve_checkpoint_and_model_type(args)
    dataset_path = Path(args.dataset)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    logger.info("Loading dataset from %s", dataset_path)
    dataset = RDD2022Dataset()
    dataset.load(dataset_path)

    logger.info("Loading model checkpoint from %s", checkpoint_path)
    model = ModelRegistry.create(model_type, {})
    model.load_checkpoint(checkpoint_path)

    # Run evaluation
    logger.info("Running evaluation...")
    engine = EvaluationEngine()
    report = engine.evaluate(model=model, dataset=dataset)

    # Save report
    output_dir = checkpoint_path.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.json"
    report.save(report_path)
    logger.info("Evaluation report saved to %s", report_path)

    # Log to experiment tracker
    tracker = ExperimentTracker(output_dir=output_dir)
    run_id = tracker.start_run(
        config={"checkpoint": str(checkpoint_path), "dataset": str(dataset_path)},
        model_name=model_type,
        dataset_name="rdd2022",
    )
    tracker.end_run(run_id, {
        "map_50": report.map_50,
        "map_50_95": report.map_50_95,
        "precision": report.precision,
        "recall": report.recall,
        "f1_score": report.f1_score,
    })

    print(f"Evaluation complete. Report saved to: {report_path}")
    print(f"  mAP@0.5:    {report.map_50:.4f}")
    print(f"  mAP@0.5:95: {report.map_50_95:.4f}")
    print(f"  Precision:   {report.precision:.4f}")
    print(f"  Recall:      {report.recall:.4f}")
    print(f"  F1-score:    {report.f1_score:.4f}")


def handle_predict(args: argparse.Namespace) -> None:
    """Handle the predict subcommand.

    Loads a model checkpoint and runs inference on input images,
    saving annotated results to the output directory.

    Args:
        args: Parsed command-line arguments.
    """
    import model.models  # noqa: F401 — registers all model wrappers
    from model.inference.pipeline import InferencePipeline
    from model.models.registry import ModelRegistry

    checkpoint_path, model_type = _resolve_checkpoint_and_model_type(args)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Load model
    logger.info("Loading model checkpoint from %s", checkpoint_path)
    model = ModelRegistry.create(model_type, {})
    model.load_checkpoint(checkpoint_path)

    # Create inference pipeline
    pipeline = InferencePipeline(model=model)

    # Run inference
    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        logger.info("Running batch inference on directory: %s", input_path)
        predictions = pipeline.predict_directory(input_path)
        for image_name, boxes in predictions.items():
            img_path = input_path / image_name
            out_file = output_path / image_name
            pipeline.save_annotated(img_path, boxes, out_file)
            logger.debug("Saved annotated: %s", out_file)
        print(f"Inference complete. Processed {len(predictions)} images.")
        print(f"Results saved to: {output_path}")
    else:
        logger.info("Running inference on image: %s", input_path)
        predictions = pipeline.predict_image(input_path)
        out_file = output_path / input_path.name
        pipeline.save_annotated(input_path, predictions, out_file)
        print(f"Inference complete. {len(predictions)} detections found.")
        print(f"Result saved to: {out_file}")


def handle_list_models(args: argparse.Namespace) -> None:
    """Handle the list-models subcommand.

    Prints all registered model identifiers to stdout.

    Args:
        args: Parsed command-line arguments.
    """
    from model.models.registry import ModelRegistry

    # Ensure model wrappers are imported so they register themselves
    import model.models.ssd_mobilenet  # noqa: F401
    import model.models.yolov6_wrapper  # noqa: F401

    models = ModelRegistry.list_models()
    if models:
        print("Registered models:")
        for name in models:
            print(f"  - {name}")
    else:
        print("No models registered.")


def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entry point.

    Parses arguments and dispatches to the appropriate handler.

    Args:
        argv: Command-line arguments. If None, uses sys.argv.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Dispatch to handler
    handlers = {
        "train": handle_train,
        "evaluate": handle_evaluate,
        "predict": handle_predict,
        "list-models": handle_list_models,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except Exception as e:
        logger.debug("Exception details:", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
