"""Command-line interface for the Road Damage Evaluation Framework.

Provides subcommands for training, evaluation, inference, and model listing.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


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

    # evaluate subcommand
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained model on a dataset.",
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint file.",
    )
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
    predict_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint file.",
    )
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

    logger.info("Starting real PyTorch training with config: %s", config_path)
    results = train(config_path=config_path, verbose=verbose)

    print(f"\nTraining complete.")
    print(f"  Final train loss: {results.get('final_train_loss', 'N/A'):.4f}")
    print(f"  Final val loss:   {results.get('final_val_loss', 'N/A'):.4f}")
    print(f"  Best val loss:    {results.get('best_val_loss', 'N/A'):.4f}")
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
    from model.datasets.rdd2022 import RDD2022Dataset
    from model.evaluation.engine import EvaluationEngine
    from model.models.registry import ModelRegistry
    from model.tracking.tracker import ExperimentTracker

    checkpoint_path = Path(args.checkpoint)
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

    # Determine model type from checkpoint metadata or use first registered model
    # For now, we attempt to load checkpoint info
    logger.info("Loading model checkpoint from %s", checkpoint_path)
    available_models = ModelRegistry.list_models()
    if not available_models:
        print("Error: No models registered.", file=sys.stderr)
        sys.exit(1)

    # Use the first available model as default (user should specify in config)
    model_type = available_models[0]
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
    from model.inference.pipeline import InferencePipeline
    from model.models.registry import ModelRegistry

    checkpoint_path = Path(args.checkpoint)
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
    available_models = ModelRegistry.list_models()
    if not available_models:
        print("Error: No models registered.", file=sys.stderr)
        sys.exit(1)

    model_type = available_models[0]
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
