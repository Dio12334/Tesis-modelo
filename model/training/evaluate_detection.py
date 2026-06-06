"""Evaluate a trained detection model on the RDD2022 dataset.

Generic evaluation script that works with any model registered in ModelRegistry
(SSD MobileNetV3, YOLO26, YOLOv6, etc.). Computes mAP, precision, recall, F1,
and confusion matrix on the validation or test set using a saved checkpoint.

Usage:
    # Using a training config file (recommended)
    python -m model.training.evaluate_detection --config model/configs/train_yolo26.yaml --checkpoint checkpoints/yolo26/best_model.pt
    
    # Using run-id (looks for checkpoint in standard location)
    python -m model.training.evaluate_detection --config model/configs/train_yolo26.yaml --run-id <uuid>
    
    # Legacy mode with manual parameters
    python -m model.training.evaluate_detection --checkpoint <path_to_pt_file> --model-type ssd_mobilenet
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image

from model.config.manager import ConfigManager
from model.datasets.rdd2022 import RDD2022Dataset
from model.evaluation.metrics import (
    compute_confusion_matrix,
    compute_map,
    compute_precision_recall_f1,
)
from model.exceptions import (
    ConfigurationError,
    DatasetNotFoundError,
    ModelNotFoundError,
    ParseError,
    ValidationError,
)
from model.models.registry import BaseDetector, ModelRegistry

logger = logging.getLogger(__name__)


def load_and_merge_config(
    config_path: Optional[str],
    overrides: Optional[dict] = None,
) -> dict:
    """Load a YAML configuration, resolve env vars, and merge CLI overrides.

    The configuration file (when provided) is parsed via ``ConfigManager.load``
    (which uses ``yaml.safe_load``), ``${ENV}`` references are substituted via
    ``ConfigManager.resolve_env_vars`` before any validation, and the supplied
    ``overrides`` are deep-merged on top so that override leaf values take
    precedence while unmentioned base values are preserved and nested dicts are
    merged recursively (rather than replaced wholesale).

    Args:
        config_path: Path to the YAML configuration file. When ``None`` an empty
            base configuration is used and only the overrides apply.
        overrides: CLI/programmatic override dict deep-merged on top of the
            loaded configuration. Override values win.

    Returns:
        The merged configuration dict (sections ``model``, ``dataset``,
        ``evaluation``, ``checkpoint``) with environment variables resolved.

    Raises:
        ConfigurationError: If the configuration file does not exist or cannot
            be parsed as YAML. The message carries the offending path and the
            underlying parser error.

    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
    """
    overrides = overrides or {}
    config_manager = ConfigManager()

    if config_path is not None:
        try:
            # Req 3.1: parse the YAML file (ConfigManager.load uses yaml.safe_load).
            base_config = config_manager.load(Path(config_path))
        except ValidationError as exc:
            # Req 3.5: translate the loader's ValidationError (missing file or
            # unparseable YAML) into a ConfigurationError. The loader's messages
            # already carry the offending path and the underlying parser error.
            raise ConfigurationError(exc.schema_violations) from exc
    else:
        base_config = {}

    # Req 3.4: substitute ${ENV} references before validation occurs.
    base_config = config_manager.resolve_env_vars(base_config)

    # Req 3.3: deep-merge overrides on top; override leaf values take precedence.
    merged = config_manager.merge(base_config, overrides)
    return merged


def _get_nested(config: dict, keys: tuple) -> tuple:
    """Safely navigate a nested mapping without any filesystem access.

    Args:
        config: The (possibly nested) configuration mapping.
        keys: An ordered tuple of keys describing the path to the value.

    Returns:
        A ``(found, value)`` pair. ``found`` is ``False`` (and ``value`` is
        ``None``) whenever any intermediate node is missing or is not a mapping;
        otherwise ``found`` is ``True`` and ``value`` is the resolved leaf.
    """
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current


def validate_config(config: dict) -> None:
    """Validate the merged configuration, collecting every violation.

    All validation rules are evaluated before terminating so that the user
    receives a single, comprehensive error rather than one failure at a time
    (Req 4.1). The function is pure: it performs **no** filesystem access, so
    the checkpoint exclusive-or is decided purely from the presence of the two
    keys without probing the disk (Req 12.3).

    Rules:
        * Required and non-null: ``model.type``, ``model.config.num_classes``,
          ``dataset.path``, ``evaluation.split`` (Req 4.2).
        * ``evaluation.confidence_threshold`` / ``evaluation.iou_threshold``,
          when provided, must lie in the closed interval ``[0.0, 1.0]``
          (Req 4.3, 4.4).
        * ``evaluation.split``, when provided, must be one of ``train``,
          ``val``, ``test`` (Req 4.5).
        * Exactly one of ``checkpoint.path`` / ``checkpoint.run_id`` must be
          present and non-null (Req 4.6, 4.7, 4.8).

    When any violation is detected, a single :class:`ConfigurationError` is
    raised whose rendered message is a header followed by one bullet line per
    violation, with each bullet naming the offending parameter and the rule it
    violated, including the observed value and the expected range/allowed
    values where applicable (Req 4.9, 11.1, 11.3, 11.4). The exception's
    ``violations`` attribute retains the raw per-violation strings.

    Args:
        config: The merged configuration dict to validate.

    Raises:
        ConfigurationError: If one or more validation rules are violated.

    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 11.1, 11.3,
        11.4, 12.3
    """
    if not isinstance(config, dict):
        config = {}

    violations: List[str] = []

    # Req 4.2: required parameters must be present and non-null.
    required_params = [
        ("model.type", ("model", "type")),
        ("model.config.num_classes", ("model", "config", "num_classes")),
        ("dataset.path", ("dataset", "path")),
        ("evaluation.split", ("evaluation", "split")),
    ]
    for name, keys in required_params:
        found, value = _get_nested(config, keys)
        if not found or value is None:
            violations.append(
                f"{name}: required parameter is missing or null; "
                f"a non-null value is required"
            )

    # Req 4.3, 4.4: thresholds, when provided, must lie in [0.0, 1.0].
    for name, keys in [
        ("evaluation.confidence_threshold", ("evaluation", "confidence_threshold")),
        ("evaluation.iou_threshold", ("evaluation", "iou_threshold")),
    ]:
        found, value = _get_nested(config, keys)
        if found and value is not None:
            # bool is a subclass of int; reject it as a non-numeric threshold.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                violations.append(
                    f"{name}: observed value {value!r} is not a number; "
                    f"expected a value in the closed interval [0.0, 1.0]"
                )
            elif not (0.0 <= float(value) <= 1.0):
                violations.append(
                    f"{name}: observed value {value} is out of range; "
                    f"expected a value in the closed interval [0.0, 1.0]"
                )

    # Req 4.5: split, when provided, must be one of the allowed values.
    allowed_splits = ["train", "val", "test"]
    found_split, split_value = _get_nested(config, ("evaluation", "split"))
    if found_split and split_value is not None and split_value not in allowed_splits:
        violations.append(
            f"evaluation.split: observed value {split_value!r} is not allowed; "
            f"expected one of {allowed_splits}"
        )

    # Req 4.6, 4.7, 4.8, 12.3: checkpoint.path XOR checkpoint.run_id. Decided
    # purely from key presence -- no filesystem lookup is performed here.
    found_path, path_value = _get_nested(config, ("checkpoint", "path"))
    found_run_id, run_id_value = _get_nested(config, ("checkpoint", "run_id"))
    has_path = found_path and path_value is not None
    has_run_id = found_run_id and run_id_value is not None
    if has_path and has_run_id:
        violations.append(
            "checkpoint.path / checkpoint.run_id: the two options are mutually "
            "exclusive; provide exactly one of them, not both"
        )
    elif not has_path and not has_run_id:
        violations.append(
            "checkpoint.path / checkpoint.run_id: one of them is required; "
            "provide exactly one of them"
        )

    if violations:
        # Req 11.3: header followed by one bullet line per violation. The
        # ConfigurationError keeps the raw violation strings on `.violations`,
        # while its rendered message is reformatted as a bulleted list.
        header = (
            "Configuration validation failed with the following "
            f"{len(violations)} error(s):"
        )
        message = header + "\n" + "\n".join(f"  - {v}" for v in violations)
        error = ConfigurationError(violations)
        error.args = (message,)
        raise error


def select_device() -> "torch.device":
    """Select the inference Device, preferring CUDA when available.

    Returns ``torch.device("cuda")`` when :func:`torch.cuda.is_available`
    returns ``True`` and ``torch.device("cpu")`` otherwise (Req 10.1). The
    selected Device is logged at INFO level so the choice is visible before the
    Detector is loaded (Req 10.2).

    Returns:
        The selected :class:`torch.device`.

    Requirements: 10.1, 10.2
    """
    # Req 10.1: cuda when available, else cpu.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Req 10.2: log the selection at INFO before the Detector is loaded.
    logger.info("Using device: %s", device)
    return device


def resolve_checkpoint(config: dict) -> Path:
    """Resolve the Checkpoint_Path from a direct path or a training Run_Id.

    Resolution is decided from the (already-validated) checkpoint section of the
    merged configuration:

    * When ``checkpoint.path`` is present and non-null, that value is returned
      verbatim as a :class:`~pathlib.Path` without probing the filesystem; any
      existence/openability check is deferred to the loading stage (Req 5.1).
    * Otherwise, when ``checkpoint.run_id`` is present, two candidate paths are
      built under ``<checkpoint_dir>/<model.type>/<run_id>/`` in search order --
      ``best_model.pt`` first, then ``last_model.pt`` -- and the first candidate
      that exists on disk is returned (Req 5.2, 5.3). ``checkpoint.checkpoint_dir``
      defaults to ``./checkpoints`` when absent.

    When ``checkpoint.run_id`` is used and neither candidate exists, a
    :class:`FileNotFoundError` is raised whose message lists every searched path
    in the order it was searched (Req 5.4, 12.1).

    This function assumes the configuration has already passed
    :func:`validate_config` (which enforces the ``checkpoint.path`` XOR
    ``checkpoint.run_id`` rule). If both are somehow present, ``checkpoint.path``
    takes precedence.

    Args:
        config: The merged, validated configuration dict.

    Returns:
        The resolved Checkpoint_Path as a :class:`~pathlib.Path`.

    Raises:
        FileNotFoundError: If resolution proceeds via ``checkpoint.run_id`` and
            none of the candidate checkpoint files exist on disk. The message
            lists every searched path in search order.

    Requirements: 5.1, 5.2, 5.3, 5.4, 12.1
    """
    if not isinstance(config, dict):
        config = {}

    # Req 5.1: a direct path is returned verbatim (no filesystem probe here).
    found_path, path_value = _get_nested(config, ("checkpoint", "path"))
    if found_path and path_value is not None:
        return Path(path_value)

    # Req 5.2, 5.3: resolve from run_id by building best_model.pt then
    # last_model.pt candidates under <checkpoint_dir>/<model.type>/<run_id>/.
    _, run_id = _get_nested(config, ("checkpoint", "run_id"))
    _, model_type = _get_nested(config, ("model", "type"))

    # checkpoint_dir defaults to ./checkpoints when absent or null.
    found_dir, checkpoint_dir = _get_nested(config, ("checkpoint", "checkpoint_dir"))
    if not found_dir or checkpoint_dir is None:
        checkpoint_dir = "./checkpoints"

    run_dir = Path(checkpoint_dir) / str(model_type) / str(run_id)
    candidates = [run_dir / "best_model.pt", run_dir / "last_model.pt"]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Req 5.4, 12.1: no candidate exists -- list every searched path in order.
    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "No checkpoint file found for run_id "
        f"{run_id!r}. Searched the following path(s) in order:\n{searched}"
    )


def load_checkpoint_into(detector, checkpoint_path: Path) -> None:
    """Load a resolved Checkpoint_Path into the Detector, translating errors.

    Delegates to ``detector.load_checkpoint(checkpoint_path)`` (Req 5.6) and
    classifies any failure into one of two categories so that the caller can
    distinguish "the file could not be opened" from "the file opened but its
    contents are corrupt or incompatible":

    * **Open failures propagate unchanged.** A :class:`FileNotFoundError`
      (the resolved path does not exist) is re-raised as-is (Req 5.5). Any other
      :class:`OSError` raised while opening the file -- for example a
      :class:`PermissionError` or a low-level I/O error -- also propagates
      unchanged and is **not** converted into a ``FileNotFoundError`` or any
      other type (Req 5.5).
    * **Corrupt/incompatible load failures become ``RuntimeError``.** Every
      other exception (e.g. a ``RuntimeError`` from ``torch.load`` on a
      truncated/corrupt archive, an ``EOFError``/unpickling error, or a
      ``KeyError``/``ValueError`` from an incompatible state dict) is re-raised
      as a :class:`RuntimeError` whose message names the Checkpoint_Path and
      includes the underlying exception text, chained via ``from`` so the
      original cause is preserved (Req 5.7, 12.2).

    Because this function either returns ``None`` only after loading has fully
    completed or raises on failure, the caller cannot begin inference until
    loading has succeeded -- a checkpoint error can never coexist with a
    concurrent inference attempt (Req 12.3).

    Args:
        detector: The Detector whose ``load_checkpoint`` method performs the
            actual deserialization and state-dict application.
        checkpoint_path: The resolved Checkpoint_Path to load.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist (propagated
            from ``detector.load_checkpoint`` unchanged).
        OSError: If the checkpoint file exists but cannot be opened due to a
            non-``FileNotFoundError`` cause such as a permission or I/O error
            (propagated unchanged).
        RuntimeError: If loading fails because the checkpoint is corrupted or
            incompatible. The message identifies the Checkpoint_Path and the
            underlying cause.

    Requirements: 5.5, 5.6, 5.7, 12.2, 12.3
    """
    try:
        # Req 5.6: pass the resolved path to Detector.load_checkpoint(path).
        detector.load_checkpoint(checkpoint_path)
    except OSError:
        # Req 5.5: open failures -- FileNotFoundError as well as other OSError
        # subclasses (PermissionError, IsADirectoryError, low-level I/O errors)
        # -- propagate unchanged. FileNotFoundError is itself an OSError, so a
        # single handler covers both without converting one type into another.
        raise
    except Exception as exc:
        # Req 5.7, 12.2: the file opened but its contents are corrupt or
        # incompatible (e.g. torch.load RuntimeError/EOFError/unpickling error,
        # or KeyError/ValueError from an incompatible state dict). Re-raise as a
        # RuntimeError naming the Checkpoint_Path and the underlying cause.
        raise RuntimeError(
            f"Failed to load checkpoint from '{checkpoint_path}': {exc}"
        ) from exc


def _levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    The edit distance is the minimum number of single-character insertions,
    deletions, or substitutions required to transform ``a`` into ``b``. Used to
    drive the "Did you mean" suggestion for an unregistered model name.

    Args:
        a: The first string.
        b: The second string.

    Returns:
        The non-negative integer edit distance between ``a`` and ``b``.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Classic two-row dynamic-programming distance computation.
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            substitute_cost = previous[j - 1] + (0 if ca == cb else 1)
            current[j] = min(insert_cost, delete_cost, substitute_cost)
        previous = current
    return previous[-1]


def _closest_model(model_type: str, available_models: List[str]) -> Optional[str]:
    """Return the closest registered name within edit distance ``[1, 2]``.

    Computes the Levenshtein distance between ``model_type`` and every available
    model name and returns the name with the smallest distance, provided that
    distance is greater than zero and at most two (Req 13.2). When no name falls
    within that band, ``None`` is returned (no suggestion). Ties are broken by
    the order of ``available_models``; passing an alphabetically sorted list
    therefore yields the alphabetically-first closest match.

    Args:
        model_type: The requested (unregistered) model name.
        available_models: The registered model names to compare against.

    Returns:
        The closest model name whose edit distance lies in ``[1, 2]``, or
        ``None`` when no such name exists.
    """
    best_name: Optional[str] = None
    best_distance: Optional[int] = None
    for name in available_models:
        distance = _levenshtein(model_type, name)
        if 1 <= distance <= 2 and (best_distance is None or distance < best_distance):
            best_name = name
            best_distance = distance
    return best_name


def build_detector(model_type: str, model_config: dict) -> "BaseDetector":
    """Instantiate a Detector through Model_Registry, enriching its errors.

    Thin wrapper around ``ModelRegistry.create(model_type, model_config)``
    (Req 1.1) that distinguishes three failure modes so the caller receives an
    actionable, model-agnostic error:

    * **Unregistered model type** (no exact match): a :class:`ModelNotFoundError`
      is raised whose message lists every registered model name in alphabetical
      order (Req 1.7, 13.1) and includes a "Did you mean: <name>?" suggestion
      when some registered name lies within edit distance ``[1, 2]`` of the
      requested value (Req 13.2). Membership is checked here so the suggestion
      can be attached before delegating to the registry.
    * **Exact-name match that fails to instantiate for a non-schema reason**
      (e.g. the registered class loaded but is unusable): a
      :class:`ModelNotFoundError` is raised carrying the underlying instantiation
      error text via ``cause`` and **no** "Did you mean" suggestion (Req 13.3).
    * **Schema validation failure**: the :class:`ConfigurationError` raised by
      the registry -- which enumerates every schema violation -- is propagated
      unchanged (Req 13.4).

    Args:
        model_type: The configured ``model.type`` (registered model name).
        model_config: The configured ``model.config`` passed verbatim to
            ``ModelRegistry.create``.

    Returns:
        The instantiated :class:`BaseDetector`.

    Raises:
        ModelNotFoundError: If ``model_type`` is not registered (with an
            alphabetical list and an optional suggestion) or if an exactly-named
            model fails to instantiate for a non-schema reason (with ``cause``
            set and no suggestion).
        ConfigurationError: If the model configuration fails Model_Registry
            schema validation (propagated, listing every violation).

    Requirements: 1.1, 1.7, 13.1, 13.2, 13.3, 13.4
    """
    # list_models() already returns names in alphabetical order (Req 13.1).
    available_models = ModelRegistry.list_models()

    # Req 1.7, 13.1, 13.2: unregistered type -> enriched ModelNotFoundError with
    # the alphabetical list plus an edit-distance suggestion when one exists.
    if model_type not in available_models:
        suggestion = _closest_model(model_type, available_models)
        raise ModelNotFoundError(
            model_type, available_models, suggestion=suggestion
        )

    try:
        # Req 1.1: instantiate exclusively through ModelRegistry.create.
        return ModelRegistry.create(model_type, model_config)
    except ConfigurationError:
        # Req 13.4: schema validation failure -- propagate unchanged so every
        # listed violation reaches the caller.
        raise
    except ModelNotFoundError:
        # Defensive: the exact name was confirmed registered above, but if the
        # registry still reports it missing, surface that unchanged.
        raise
    except Exception as exc:
        # Req 13.3: exact-name match whose instantiation failed for a non-schema
        # reason. Attach the underlying cause and OMIT any suggestion.
        raise ModelNotFoundError(
            model_type, available_models, cause=str(exc)
        ) from exc


def _load_dataset(dataset: "RDD2022Dataset", path: Path) -> None:
    """Load ``dataset`` from ``path``, translating loader exceptions.

    Delegates to ``dataset.load(path)`` and converts the dataset layer's
    exceptions into the script-boundary types required by the requirements:

    * :class:`DatasetNotFoundError` (the path disappeared between the up-front
      existence check and the load, or a required subset directory is absent)
      becomes a :class:`FileNotFoundError` whose message names the offending
      path (Req 14.1).
    * :class:`ParseError` (a malformed annotation file) becomes a
      :class:`ConfigurationError` whose single violation names the annotation
      file and the underlying parser error (Req 14.3).

    Args:
        dataset: The :class:`RDD2022Dataset` instance to populate.
        path: The dataset root directory to load from.

    Raises:
        FileNotFoundError: If the dataset path/subset is missing.
        ConfigurationError: If an annotation file fails to parse.
    """
    try:
        dataset.load(path)
    except DatasetNotFoundError as exc:
        # Req 14.1: translate the dataset layer's "not found" into the
        # script-boundary FileNotFoundError, naming the offending path.
        raise FileNotFoundError(f"Dataset path not found: {exc.path}") from exc
    except ParseError as exc:
        # Req 14.3: a malformed annotation file becomes a ConfigurationError
        # naming the annotation file and the parser error.
        violation = (
            f"annotation file '{exc.file_path}' failed to parse "
            f"(line {exc.line_number}): {exc.description}"
        )
        raise ConfigurationError([violation]) from exc


def load_split(
    config: dict,
) -> Tuple["RDD2022Dataset", List[str], Dict[int, str], List[str]]:
    """Load the dataset and produce the requested evaluation partition.

    The dataset path is taken from ``dataset.path`` and its existence is
    verified up front so that a missing dataset fails fast -- before any
    inference begins -- with a :class:`FileNotFoundError` naming the path
    (Req 14.1). The dataset is then loaded through :func:`_load_dataset`, which
    translates a malformed annotation file into a :class:`ConfigurationError`
    naming the annotation file and the parser error (Req 14.3).

    The partition is selected from ``evaluation.split`` (already constrained to
    ``train``/``val``/``test`` by :func:`validate_config`):

    * ``test`` loads the dataset's ``test`` subset directly and evaluates on it
      in full (Req 9.3).
    * ``train`` / ``val`` load the ``train`` subset and split it into
      train/val/(test) partitions via :meth:`RDD2022Dataset.split`, using
      ``evaluation.val_split`` (default ``0.2``) for the validation fraction and
      ``evaluation.seed`` (default ``42``) for a reproducible shuffle; the
      ``train`` or ``val`` partition is then returned (Req 9.1, 9.2).

    Class names are derived from the loaded source dataset (the full train
    subset for ``train``/``val`` so that classes present only in the validation
    partition are still represented, and the test subset for ``test``). The
    ``idx_to_class`` mapping assigns one-based indices to the sorted class names
    (index ``0`` is reserved for background), matching the label indexing used
    throughout the evaluation pipeline.

    Args:
        config: The merged, validated configuration dict.

    Returns:
        A ``(split_dataset, class_names, idx_to_class, display_class_names)``
        tuple where ``split_dataset`` is the :class:`RDD2022Dataset` partition
        to evaluate, ``class_names`` is the sorted list of raw English
        class-name strings (the canonical training index space),
        ``idx_to_class`` maps each label index to its English class name, and
        ``display_class_names`` is a parallel list whose ``i``-th entry is the
        Spanish display name for ``class_names[i]`` when a ``class_mapping``
        config is provided (otherwise ``display_class_names == class_names``).

    Raises:
        FileNotFoundError: If ``dataset.path`` does not exist (raised before
            inference begins) or a required subset directory is missing.
        ConfigurationError: If an annotation file fails to parse during loading.

    Requirements: 9.1, 9.2, 9.3, 14.1, 14.3
    """
    if not isinstance(config, dict):
        config = {}

    # Resolve dataset path and verify it exists up front (Req 14.1).
    _, dataset_path = _get_nested(config, ("dataset", "path"))
    path = Path(str(dataset_path))
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    # Optional dataset/evaluation knobs with documented defaults.
    _, country_filter = _get_nested(config, ("dataset", "country_filter"))

    _, split = _get_nested(config, ("evaluation", "split"))

    found_val_split, val_split = _get_nested(config, ("evaluation", "val_split"))
    if not found_val_split or val_split is None:
        val_split = 0.2

    found_seed, seed = _get_nested(config, ("evaluation", "seed"))
    if not found_seed or seed is None:
        seed = 42

    if split == "test":
        # Req 9.3: evaluate the dataset's test subset in full.
        dataset = RDD2022Dataset(country_filter=country_filter, subset="test")
        _load_dataset(dataset, path)
        split_dataset = dataset
        logger.info("Evaluating on TEST set: %d images", len(split_dataset))
    else:
        # Req 9.1, 9.2: load the train subset and partition it reproducibly.
        dataset = RDD2022Dataset(country_filter=country_filter, subset="train")
        _load_dataset(dataset, path)
        train_ratio = 1.0 - float(val_split)
        train_ds, val_ds, _ = dataset.split(
            train_ratio, float(val_split), 0.0, seed=int(seed)
        )
        if split == "train":
            split_dataset = train_ds
            logger.info("Evaluating on TRAIN set: %d images", len(split_dataset))
        else:
            split_dataset = val_ds
            logger.info("Evaluating on VALIDATION set: %d images", len(split_dataset))

    # Class names come from the loaded source dataset; idx_to_class uses
    # one-based indices (index 0 is reserved for background).
    class_names = dataset.get_class_names()

    # Fallback: if no class names were found in annotations (e.g., blind test
    # set with no ground-truth labels), read them from meta.json which defines
    # the project's class schema.
    if not class_names:
        meta_path = path / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta_classes = meta.get("classes", [])
                class_names = sorted(c["title"] for c in meta_classes if "title" in c)
                if class_names:
                    logger.info(
                        "No class names in annotations; loaded %d classes from meta.json",
                        len(class_names),
                    )
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    # The canonical class-index space is the sorted list of raw English
    # labels produced by RDD2022Dataset.get_class_names(); this is identical
    # to the index space the training pipeline builds in
    # train_detection.py:RDD2022TorchDataset._class_to_idx, so predictions
    # emitted by the trained detector are positionally aligned with these
    # class names. We must NOT override class_names with the YAML taxonomy
    # (Spanish, declared order) -- doing so would scramble every metric.
    #
    # When a class_mapping config is provided, the YAML taxonomy is used
    # purely for display rendering: a parallel display_class_names list is
    # produced via TargetMapper.map_class so downstream consumers can render
    # Spanish labels on confusion-matrix axes / report tables without
    # disturbing the canonical (training-aligned) index space.
    _, class_mapping_path = _get_nested(config, ("dataset", "class_mapping"))
    display_class_names: List[str] = list(class_names)
    if class_mapping_path and Path(str(class_mapping_path)).exists():
        from model.datasets.target_mapper import TargetMapper
        target_mapper = TargetMapper(Path(str(class_mapping_path)), strict=False)
        if target_mapper.taxonomy:
            mapped: List[str] = []
            for name in class_names:
                try:
                    mapped.append(target_mapper.map_class(name))
                except Exception:
                    # Fall back to the canonical English name when mapping
                    # fails for a particular raw label; logged at WARNING so
                    # the gap is visible without aborting evaluation.
                    logger.warning(
                        "TargetMapper has no entry for raw label %r; "
                        "using canonical name in display_class_names",
                        name,
                    )
                    mapped.append(name)
            display_class_names = mapped
            logger.info(
                "Display class names derived from class_mapping (%d classes): "
                "english=%s display=%s",
                len(class_names),
                class_names,
                display_class_names,
            )

    idx_to_class = {idx: name for idx, name in enumerate(class_names)}

    return split_dataset, class_names, idx_to_class, display_class_names


def normalize_box(
    box: List[float], input_size: int
) -> Tuple[List[float], str]:
    """Normalize a single detector output box, reporting the detected mode.

    A box is treated as Pixel_Coordinates when **any** of its four coordinates
    exceeds ``1.0``; in that case every coordinate is divided by the inference
    ``input_size`` used for that image (Req 6.1) and the returned mode is
    ``"pixel"``. Otherwise every coordinate already lies within ``[0.0, 1.0]``
    and the values are passed through unchanged (Req 6.2) with mode
    ``"normalized"``. The returned mode lets the caller emit the per-batch
    DEBUG normalization-decision log required by Req 6.3.

    This function performs the scale conversion only; clamping out-of-range
    values into ``[0, 1]`` and dropping degenerate boxes are the
    responsibility of :func:`clamp_and_filter` (Req 6.4, 6.5).

    Args:
        box: The detector output box as ``[x_min, y_min, x_max, y_max]``.
        input_size: The positive inference input size (image side length in
            pixels) used for the image the box came from.

    Returns:
        A ``(normalized_box, mode)`` pair where ``normalized_box`` is the
        four-element coordinate list and ``mode`` is ``"pixel"`` or
        ``"normalized"``.

    Requirements: 6.1, 6.2
    """
    x_min, y_min, x_max, y_max = box

    # Req 6.1: any coordinate above 1.0 marks the box as pixel-space.
    if x_min > 1.0 or y_min > 1.0 or x_max > 1.0 or y_max > 1.0:
        normalized = [
            x_min / input_size,
            y_min / input_size,
            x_max / input_size,
            y_max / input_size,
        ]
        return normalized, "pixel"

    # Req 6.2: already normalized -- pass the values through unchanged.
    return [x_min, y_min, x_max, y_max], "normalized"


def clamp_and_filter(
    box: List[float], image_id: str
) -> Optional[List[float]]:
    """Clamp a normalized box into ``[0, 1]`` and drop degenerate boxes.

    Each coordinate that falls outside the closed interval ``[0.0, 1.0]`` is
    clamped into it, and a WARNING identifying the affected image and the
    original (pre-clamp) value is logged for every coordinate that had to be
    adjusted (Req 6.4). After clamping, a box whose ``x_min > x_max`` or
    ``y_min > y_max`` is degenerate; such a box is excluded from metrics by
    returning ``None``, again after logging a WARNING identifying the image
    (Req 6.5).

    The degeneracy test is applied to the clamped coordinates, so a box that
    becomes degenerate only as a consequence of clamping is still excluded.

    Args:
        box: The (already scale-normalized) box as
            ``[x_min, y_min, x_max, y_max]``.
        image_id: Identifier of the image the box belongs to, included in any
            warning so the affected image can be located.

    Returns:
        The clamped ``[x_min, y_min, x_max, y_max]`` list when the box is
        non-degenerate, or ``None`` when the box is degenerate and must be
        excluded.

    Requirements: 6.4, 6.5
    """
    coord_names = ("x_min", "y_min", "x_max", "y_max")
    clamped: List[float] = []
    for name, value in zip(coord_names, box):
        if value < 0.0:
            # Req 6.4: clamp below-range value, warning with image + original.
            logger.debug(
                "Image %s: %s coordinate %s is below 0.0; clamping to 0.0",
                image_id,
                name,
                value,
            )
            clamped.append(0.0)
        elif value > 1.0:
            # Req 6.4: clamp above-range value, warning with image + original.
            logger.debug(
                "Image %s: %s coordinate %s exceeds 1.0; clamping to 1.0",
                image_id,
                name,
                value,
            )
            clamped.append(1.0)
        else:
            clamped.append(value)

    x_min, y_min, x_max, y_max = clamped

    # Req 6.5: exclude degenerate boxes (after clamping) with a warning.
    if x_min > x_max or y_min > y_max:
        logger.warning(
            "Image %s: degenerate box %s excluded "
            "(x_min > x_max or y_min > y_max after normalization)",
            image_id,
            clamped,
        )
        return None

    return clamped


def map_label(index: int, idx_to_class: Dict[int, str]) -> str:
    """Map an integer label index to its class-name string.

    Returns the class name registered for ``index`` in ``idx_to_class`` when
    present, and falls back to the literal ``"class_<index>"`` when no mapping
    exists for that index (Req 7.4).

    Args:
        index: The integer label index produced by the Detector.
        idx_to_class: Mapping from label index to class-name string.

    Returns:
        The mapped class name, or ``"class_<index>"`` when ``index`` is absent
        from ``idx_to_class``.

    Requirements: 7.4
    """
    # Req 7.4: registered class name, else the literal class_<index> fallback.
    return idx_to_class.get(index, f"class_{index}")


def _as_float(value) -> float:
    """Coerce a coordinate/score value to a Python float.

    Detector outputs are tensors per the Base_Detector contract (Req 2.4), so a
    zero-dimensional tensor element exposes ``.item()``. The helper falls back to
    ``float(value)`` for plain numbers so the inference loop also tolerates
    detectors (or test doubles) that return Python scalars.

    Args:
        value: A tensor element or plain number.

    Returns:
        The value as a Python ``float``.
    """
    item = getattr(value, "item", None)
    if callable(item):
        return float(item())
    return float(value)


def _build_ground_truth(annotation, image_id: str, input_size: int) -> dict:
    """Build the normalized ground-truth entry for a single annotation.

    Produces exactly one ground-truth entry per annotation (Req 7.1). Each
    annotation box is passed through :func:`normalize_box` and
    :func:`clamp_and_filter` so that every coordinate entering metrics lies in
    ``[0.0, 1.0]`` and degenerate boxes are excluded (Req 6.4, 6.5); the
    corresponding class label is dropped alongside any excluded box to keep
    boxes and labels aligned.

    Args:
        annotation: The dataset :class:`~model.datasets.base.Annotation`.
        image_id: The image identifier (``str(annotation.image_path)``).
        input_size: The inference input size used for scale normalization.

    Returns:
        A ground-truth dict with ``image_id``, normalized ``boxes``, and
        class-name ``labels``. Labels are the raw English class names
        emitted by the dataset and live in the canonical (training-aligned)
        index space; no remapping is applied.
    """
    gt_boxes: List[List[float]] = []
    gt_labels: List[str] = []
    for bbox in annotation.bounding_boxes:
        raw = [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]
        normalized, _mode = normalize_box(raw, input_size)
        clamped = clamp_and_filter(normalized, image_id)
        if clamped is None:
            continue
        gt_boxes.append(clamped)
        gt_labels.append(bbox.class_label)
    return {"image_id": image_id, "boxes": gt_boxes, "labels": gt_labels}


def _empty_prediction(image_id: str) -> dict:
    """Return an empty prediction entry for a failed image.

    A failed image (image-decode failure or forward-pass exception) occupies its
    aligned slot with an empty prediction so predictions and ground truths stay
    1:1 (Req 7.3, 14.2, 15.1). The empty tensors-as-lists also let downstream
    metrics distinguish a missing prediction from a successful no-detection
    result.

    Args:
        image_id: The image identifier the empty entry stands in for.

    Returns:
        A prediction dict with empty ``boxes``, ``labels``, and ``scores``.
    """
    return {"image_id": image_id, "boxes": [], "labels": [], "scores": []}


def _build_prediction(
    output: dict,
    image_id: str,
    input_size: int,
    idx_to_class: Dict[int, str],
) -> dict:
    """Assemble a normalized prediction entry from one detector output dict.

    For every detected box: the raw coordinates are scale-normalized via
    :func:`normalize_box` (pixel boxes divided by ``input_size``, normalized
    boxes passed through -- Req 6.1, 6.2), the per-image normalization decision
    is logged at DEBUG (Req 6.3), out-of-range coordinates are clamped into
    ``[0.0, 1.0]`` and degenerate boxes are dropped via :func:`clamp_and_filter`
    (Req 6.4, 6.5), and the integer label index is mapped to a class name via
    :func:`map_label` (Req 7.4). A box dropped as degenerate also drops its label
    and score so the three lists stay aligned.

    Args:
        output: One element of the detector's ``forward`` output, a dict with
            ``boxes``, ``labels``, and ``scores`` (Req 2.3, 2.4).
        image_id: The image identifier for the prediction entry.
        input_size: The inference input size used for scale normalization.
        idx_to_class: Mapping from label index to class-name string.

    Returns:
        A prediction dict with ``image_id``, normalized ``boxes``, class-name
        ``labels``, and ``scores``.
    """
    boxes = output["boxes"]
    labels = output["labels"]
    scores = output["scores"]

    pred_boxes: List[List[float]] = []
    pred_labels: List[str] = []
    pred_scores: List[float] = []
    modes_seen = set()

    for j in range(len(boxes)):
        box = boxes[j]
        raw = [
            _as_float(box[0]),
            _as_float(box[1]),
            _as_float(box[2]),
            _as_float(box[3]),
        ]
        # Req 6.1, 6.2: scale-normalize, remembering the detected mode.
        normalized, mode = normalize_box(raw, input_size)
        modes_seen.add(mode)
        # Req 6.4, 6.5: clamp into [0, 1] and drop degenerate boxes.
        clamped = clamp_and_filter(normalized, image_id)
        if clamped is None:
            continue
        pred_boxes.append(clamped)
        # Req 7.4: map the integer label index to its class name.
        pred_labels.append(map_label(int(_as_float(labels[j])), idx_to_class))
        pred_scores.append(_as_float(scores[j]))

    # Req 6.3: record the per-image normalization decision at DEBUG level.
    if modes_seen:
        logger.debug(
            "Image %s: coordinate normalization detected %s",
            image_id,
            ", ".join(sorted(modes_seen)),
        )

    return {
        "image_id": image_id,
        "boxes": pred_boxes,
        "labels": pred_labels,
        "scores": pred_scores,
    }


def run_inference(
    detector,
    split_ds,
    device,
    input_size: int,
    idx_to_class: Dict[int, str],
) -> Tuple[List[dict], List[dict], List[str]]:
    """Run inference over a split, keeping predictions and GTs 1:1 aligned.

    Iterates the split's annotations in order and builds exactly one ground-truth
    entry and exactly one prediction entry per annotation, so that
    ``len(predictions) == len(ground_truths)`` and
    ``predictions[i]["image_id"] == ground_truths[i]["image_id"]`` for every
    ``i`` regardless of per-image outcome (Req 7.1, 7.2).

    For each image the loop:

    * loads and transforms the image (resize to ``input_size`` then to-tensor),
    * moves the input tensor to ``device`` **before** the forward pass (Req 10.4),
    * runs ``detector.forward(...)`` inside a :func:`torch.no_grad` context
      (Req 7.6),
    * scale-normalizes, clamps, and filters predicted boxes and maps label
      indices to class names (Req 6.1-6.5, 7.4).

    Ground-truth boxes are likewise normalized/clamped so that every coordinate
    entering metrics lies in ``[0.0, 1.0]`` (Req 6.4).

    Failure handling preserves partial results and 1:1 alignment:

    * An image that cannot be loaded or decoded is logged at WARNING with its
      identifier and the underlying exception, contributes an empty prediction
      entry, and records ``"<image_id>: <text>"`` in the error list (Req 14.2,
      15.2).
    * An exception raised by ``detector.forward(...)`` (or while post-processing
      its output) is logged at ERROR with the image identifier and exception
      text, contributes an empty prediction entry, and records
      ``"<image_id>: <text>"`` in the error list (Req 15.1, 15.2).

    Progress is logged at INFO after processing counts that are positive
    multiples of 50 (Req 7.5).

    Args:
        detector: The Detector exposing ``forward`` (Base_Detector contract).
        split_ds: The evaluation-split dataset exposing ``get_annotations()``.
        device: The :class:`torch.device` inputs are moved to before inference.
        input_size: The inference input size (image side length in pixels).
        idx_to_class: Mapping from label index to class-name string.

    Returns:
        A ``(predictions, ground_truths, errors)`` tuple. ``predictions`` and
        ``ground_truths`` are aligned 1:1 with the split's annotations and
        ``errors`` holds one ``"<image_id>: <text>"`` string per failed image.

    Requirements: 6.3, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 10.4, 14.2, 15.1, 15.2,
        15.3
    """
    transform = T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
        ]
    )

    annotations = split_ds.get_annotations()
    total = len(annotations)
    logger.info("Running inference on %d images...", total)

    predictions: List[dict] = []
    ground_truths: List[dict] = []
    errors: List[str] = []

    for i, annotation in enumerate(annotations):
        image_id = str(annotation.image_path)

        # Req 7.1: exactly one ground-truth entry per annotation, in order.
        ground_truths.append(_build_ground_truth(annotation, image_id, input_size))

        pred_entry: Optional[dict] = None

        # Stage 1: load + transform + move tensor to device (Req 10.4).
        try:
            image = Image.open(annotation.image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)
        except Exception as exc:
            # Req 14.2: image cannot be loaded/decoded -> WARNING + empty entry.
            logger.warning(
                "Could not load/decode image %s: %s", image_id, exc
            )
            errors.append(f"{image_id}: {exc}")
            pred_entry = _empty_prediction(image_id)

        # Stage 2: forward pass under no_grad + post-process (Req 7.6).
        if pred_entry is None:
            try:
                with torch.no_grad():
                    outputs = detector.forward(image_tensor)
                pred_entry = _build_prediction(
                    outputs[0], image_id, input_size, idx_to_class
                )
            except Exception as exc:
                # Req 15.1: forward-pass failure -> ERROR + empty entry.
                logger.error(
                    "Inference failed for image %s: %s", image_id, exc
                )
                errors.append(f"{image_id}: {exc}")
                pred_entry = _empty_prediction(image_id)

        # Req 7.2, 7.3: exactly one prediction entry per annotation.
        predictions.append(pred_entry)

        # Req 7.5: log progress at positive multiples of 50.
        processed = i + 1
        if processed % 50 == 0:
            logger.info("  Processed %d/%d images", processed, total)

    return predictions, ground_truths, errors


def _has_empty_prediction(predictions: List[dict], errors: Optional[List[str]] = None) -> bool:
    """Return ``True`` when any prediction entry is empty due to an inference error.

    An "empty prediction entry" in the context of Req 8.5 refers to entries
    produced by inference failures (image decode errors, forward-pass exceptions),
    not images where the model legitimately detected nothing above the confidence
    threshold. When an ``errors`` list is provided, this function returns ``True``
    only if there are actual inference errors. When ``errors`` is not provided
    (backward compatibility), it falls back to checking for any empty entry.

    Args:
        predictions: The aligned per-image prediction entries.
        errors: The list of inference error strings (one per failed image).

    Returns:
        ``True`` if there are inference errors that produced empty entries;
        ``False`` otherwise.
    """
    # If errors list is provided, use it as the authoritative source.
    if errors is not None:
        return len(errors) > 0

    # Fallback for backward compatibility: check for any empty entry.
    for pred in predictions:
        if (
            len(pred.get("boxes", [])) == 0
            and len(pred.get("labels", [])) == 0
            and len(pred.get("scores", [])) == 0
        ):
            return True
    return False


def compute_all_metrics(
    predictions: List[dict],
    ground_truths: List[dict],
    class_names: List[str],
    confidence_threshold: float,
    iou_threshold: float,
    errors: Optional[List[str]] = None,
) -> dict:
    """Compute the full detection metric suite for an evaluation run.

    Delegates to the already-tested :mod:`model.evaluation.metrics` collaborators
    and normalizes their key names into the snake_case fields the report
    assembly stage consumes:

    * :func:`compute_map` yields ``map_50`` and ``map_50_95`` (mAP at IoU ``0.5``
      and averaged over IoU ``0.5:0.95``) plus a ``per_class_ap`` entry for every
      class name (Req 8.1, 8.4).
    * :func:`compute_precision_recall_f1` yields overall ``precision``,
      ``recall``, and ``f1`` (surfaced here as ``f1_score``) at the configured
      ``confidence_threshold`` / ``iou_threshold`` (Req 8.2).
    * :func:`compute_confusion_matrix` yields a ``(C, C)`` confusion matrix over
      ``class_names`` at the same thresholds (Req 8.3).

    Req 8.5: when **any** image in the run has an empty prediction entry (no
    boxes, labels, or scores), the scalar metrics ``map_50``, ``map_50_95``,
    ``precision``, ``recall``, and ``f1_score`` are forced to ``0.0`` for the
    run, while the confusion matrix is still computed and returned sized to the
    number of classes. ``per_class_ap`` is left as computed (the zeroing rule
    enumerates only the five scalar fields).

    Args:
        predictions: The aligned per-image prediction entries (``labels`` are
            class-name strings; ``boxes`` are normalized ``[0, 1]``).
        ground_truths: The aligned per-image ground-truth entries.
        class_names: Ordered class names defining the confusion-matrix axes and
            the per-class-AP keys.
        confidence_threshold: Minimum confidence for precision/recall/F1 and the
            confusion matrix.
        iou_threshold: IoU threshold for matching predictions to ground truths.

    Returns:
        A metrics dict with keys ``map_50``, ``map_50_95``, ``per_class_ap``,
        ``precision``, ``recall``, ``f1_score``, and ``confusion_matrix`` (the
        latter a numpy array of shape ``(C, C)`` where ``C == len(class_names)``).

    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
    """
    # Req 8.1, 8.4: mAP@0.5, mAP@0.5:0.95, and per-class AP@0.5.
    map_results = compute_map(
        predictions=predictions,
        ground_truths=ground_truths,
        class_names=class_names,
    )

    # Req 8.2: overall precision/recall/F1 at the configured thresholds.
    prf1 = compute_precision_recall_f1(
        predictions=predictions,
        ground_truths=ground_truths,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )

    # Req 8.3: confusion matrix over the configured class names; always (C, C).
    confusion_matrix = compute_confusion_matrix(
        predictions=predictions,
        ground_truths=ground_truths,
        class_names=class_names,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
    )

    # Req 8.5: any empty prediction entry due to inference errors zeroes the
    # five scalar metrics while the (C, C) confusion matrix above is still
    # computed and returned.
    if _has_empty_prediction(predictions, errors):
        logger.info(
            "At least one image has an empty prediction entry; forcing "
            "map_50, map_50_95, precision, recall, and f1_score to 0.0"
        )
        map_50 = 0.0
        map_50_95 = 0.0
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
    else:
        map_50 = map_results["map_50"]
        map_50_95 = map_results["map_50_95"]
        precision = prf1["precision"]
        recall = prf1["recall"]
        f1_score = prf1["f1"]

    return {
        "map_50": map_50,
        "map_50_95": map_50_95,
        "per_class_ap": map_results["per_class_ap"],
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": confusion_matrix,
    }


def assemble_report(
    checkpoint_path: str,
    model_type: str,
    model_config: dict,
    dataset_path: str,
    split: str,
    num_images: int,
    num_classes: int,
    class_names: List[str],
    confidence_threshold: float,
    iou_threshold: float,
    metrics: dict,
    confusion_matrix: List[List[int]],
    errors: List[str],
    display_class_names: Optional[List[str]] = None,
) -> dict:
    """Assemble the Evaluation_Report dict, validating required metric fields.

    Builds the complete report structure with all required top-level fields and
    a ``metrics`` object containing both the new snake_case keys (``map_50``,
    ``map_50_95``, ``per_class_ap``) and the retained prior display keys
    (``mAP@0.5``, ``mAP@0.5:0.95``) for backward compatibility (Req 17.4).

    Before returning, the function validates that all required metric fields are
    present in the ``metrics`` dict. If any required field is missing, a
    :class:`RuntimeError` is raised that names every missing field, and the
    evaluation run is considered failed (Req 8.7).

    Required metric fields (Req 8.7):
        - ``map_50``
        - ``map_50_95``
        - ``precision``
        - ``recall``
        - ``f1_score``
        - ``per_class_ap``

    Args:
        checkpoint_path: The resolved Checkpoint_Path as a string.
        model_type: The configured ``model.type`` (registered model name).
        model_config: The configured ``model.config`` dict.
        dataset_path: The configured ``dataset.path``.
        split: The resolved ``evaluation.split`` value (Req 9.4).
        num_images: The number of images in the evaluation split.
        num_classes: The number of classes.
        class_names: The ordered list of class-name strings.
        confidence_threshold: The configured ``evaluation.confidence_threshold``.
        iou_threshold: The configured ``evaluation.iou_threshold``.
        metrics: The metrics dict from :func:`compute_all_metrics` containing
            ``map_50``, ``map_50_95``, ``precision``, ``recall``, ``f1_score``,
            ``per_class_ap``, and ``confusion_matrix``.
        confusion_matrix: The ``(C, C)`` confusion matrix as a nested list.
        errors: The list of error strings, each formatted as
            ``<image_id>: <exception text>``.

    Returns:
        The assembled Evaluation_Report dict with all required fields.

    Raises:
        RuntimeError: If any required metric field is missing from ``metrics``.
            The message names every missing field and marks the run as failed.

    Requirements: 8.6, 8.7, 9.4, 16.2, 17.4
    """
    # Req 8.7: validate that all required metric fields are present.
    required_metric_fields = [
        "map_50",
        "map_50_95",
        "precision",
        "recall",
        "f1_score",
        "per_class_ap",
    ]
    missing_fields = [
        field for field in required_metric_fields if field not in metrics
    ]
    if missing_fields:
        missing_str = ", ".join(f"'{f}'" for f in missing_fields)
        raise RuntimeError(
            f"Evaluation run failed: missing required metric field(s): {missing_str}"
        )

    # Req 8.6, 16.2: build the metrics object with both new snake_case keys and
    # retained prior display keys (Req 17.4).
    report_metrics = {
        # New snake_case keys (Req 8.6)
        "map_50": metrics["map_50"],
        "map_50_95": metrics["map_50_95"],
        "per_class_ap": metrics["per_class_ap"],
        # Prior display keys retained for backward compatibility (Req 17.4)
        "mAP@0.5": metrics["map_50"],
        "mAP@0.5:0.95": metrics["map_50_95"],
        # Standard metric fields
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
    }

    # Req 16.2: build the report with all required top-level fields.
    # Req 9.4: include the resolved evaluation.split value.
    # Req 15.3: include the error count and error list.
    report = {
        "checkpoint": checkpoint_path,
        "model_type": model_type,
        "model_config": model_config,
        "dataset": dataset_path,
        "split": split,
        "num_images": num_images,
        "num_classes": num_classes,
        "class_names": class_names,
        "display_class_names": (
            list(display_class_names)
            if display_class_names is not None
            else list(class_names)
        ),
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "metrics": report_metrics,
        "confusion_matrix": confusion_matrix,
        "errors": {
            "count": len(errors),
            "items": errors,
        },
    }

    return report


def write_outputs(
    report: dict,
    predictions: List[dict],
    ground_truths: List[dict],
    output_dir: Optional[str],
    split: str,
    checkpoint_path: Path,
) -> Tuple[Path, Path]:
    """Write the evaluation report and per-image predictions to JSON files.

    Resolves the output directory from ``output_dir`` (when non-null) or the
    parent directory of the resolved ``checkpoint_path`` (Req 16.4, 16.5).
    Writes two split-tagged JSON files so that train/val/test runs do not
    overwrite each other (Req 9.5, 16.6):

    * ``<split>_evaluation_report.json`` — the full Evaluation_Report (Req 16.1).
    * ``<split>_inference.json`` — per-image predictions with ground truths
      (Req 16.3).

    After writing, the absolute paths of both files are logged at INFO level
    (Req 16.7).

    Output file naming:

    | split | report filename               | predictions filename     |
    |-------|-------------------------------|--------------------------|
    | train | train_evaluation_report.json  | train_inference.json     |
    | val   | val_evaluation_report.json    | val_inference.json       |
    | test  | test_evaluation_report.json   | test_inference.json      |

    The predictions file structure follows the dashboard-compatible format:

    .. code-block:: python

        {
            "checkpoint": str,
            "model_type": str,
            "dataset": str,
            "confidence_threshold": float,
            "class_names": list[str],
            "images": [
                {
                    "image_id": str,
                    "ground_truth": {"boxes": [...], "labels": [...]},
                    "predictions": {"boxes": [...], "labels": [...], "scores": [...]},
                },
                ...
            ],
        }

    Args:
        report: The assembled Evaluation_Report dict from :func:`assemble_report`.
        predictions: The aligned per-image prediction entries.
        ground_truths: The aligned per-image ground-truth entries.
        output_dir: The configured ``evaluation.output_dir``. When ``None``, the
            checkpoint's parent directory is used.
        split: The resolved ``evaluation.split`` value (``train``, ``val``, or
            ``test``).
        checkpoint_path: The resolved Checkpoint_Path, used to derive the output
            directory when ``output_dir`` is ``None``.

    Returns:
        A ``(report_path, predictions_path)`` tuple of absolute :class:`Path`
        objects pointing to the written files.

    Requirements: 9.5, 16.1, 16.3, 16.4, 16.5, 16.6, 16.7
    """
    # Req 16.4, 16.5: resolve output directory.
    if output_dir is not None:
        resolved_output_dir = Path(output_dir)
    else:
        resolved_output_dir = checkpoint_path.parent

    # Ensure the output directory exists.
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    # Req 9.5, 16.6: split-tagged filenames.
    report_filename = f"{split}_evaluation_report.json"
    predictions_filename = f"{split}_inference.json"

    report_path = resolved_output_dir / report_filename
    predictions_path = resolved_output_dir / predictions_filename

    # Req 16.1: write the Evaluation_Report as a JSON document.
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Req 16.3: write per-image predictions JSON file.
    predictions_output = {
        "checkpoint": report["checkpoint"],
        "model_type": report["model_type"],
        "dataset": report["dataset"],
        "confidence_threshold": report["confidence_threshold"],
        "class_names": report["class_names"],
        "display_class_names": report.get(
            "display_class_names", report["class_names"]
        ),
        "images": [],
    }
    for pred, gt in zip(predictions, ground_truths):
        predictions_output["images"].append({
            "image_id": pred["image_id"],
            "ground_truth": {
                "boxes": gt["boxes"],
                "labels": gt["labels"],
            },
            "predictions": {
                "boxes": pred["boxes"],
                "labels": pred["labels"],
                "scores": pred["scores"],
            },
        })

    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions_output, f, indent=2)

    # Req 16.7: log absolute paths at INFO level.
    logger.info("Evaluation report saved to: %s", report_path.resolve())
    logger.info("Per-image predictions saved to: %s", predictions_path.resolve())

    return report_path.resolve(), predictions_path.resolve()


def print_summary(
    model_type: str,
    split: str,
    num_images: int,
    metrics: dict,
) -> None:
    """Print a formatted evaluation summary to standard output.

    Prints a human-readable summary including the model type, split, number of
    images, and the five key metrics: ``map_50``, ``map_50_95``, ``precision``,
    ``recall``, and ``f1_score`` (Req 16.8).

    The summary is printed to stdout (not logged) so it appears prominently in
    the terminal regardless of logging configuration.

    Args:
        model_type: The configured ``model.type`` (registered model name).
        split: The resolved ``evaluation.split`` value.
        num_images: The number of images in the evaluation split.
        metrics: The metrics dict containing ``map_50``, ``map_50_95``,
            ``precision``, ``recall``, and ``f1_score``.

    Requirements: 16.8
    """
    # Req 16.8: print formatted summary to stdout.
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Model Type:    {model_type}")
    print(f"  Split:         {split}")
    print(f"  Num Images:    {num_images}")
    print("-" * 60)
    print(f"  mAP@0.5:       {metrics['map_50']:.4f}")
    print(f"  mAP@0.5:0.95:  {metrics['map_50_95']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print(f"  F1-score:      {metrics['f1_score']:.4f}")
    print("=" * 60 + "\n")


def evaluate(
    config_path: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> dict:
    """Orchestrate the full evaluation pipeline, wiring all stages together.

    This function sequences the evaluation pipeline stages in order:

    1. **load_and_merge_config** — Load YAML, resolve env vars, merge overrides.
    2. **validate_config** — Collect all violations before any I/O (Req 4.1).
    3. **select_device** — Choose cuda/cpu, log at INFO (Req 10.1, 10.2).
    4. **resolve_checkpoint** — Direct path or run_id with fallback (Req 5.1-5.4).
    5. **build_detector** — Instantiate via ModelRegistry (Req 1.1).
    6. **load_checkpoint_into + set_eval_mode + to_device** — Load weights, set
       eval mode, move to device exactly once (Req 1.4, 1.5, 1.6, 10.3).
    7. **load_split** — Load dataset and produce the requested partition (Req 9).
    8. **run_inference** — Forward pass with 1:1 alignment (Req 7).
    9. **compute_all_metrics** — mAP, precision, recall, F1, confusion matrix.
    10. **assemble_report** — Build report, validate required fields (Req 8.6, 8.7).
    11. **write_outputs** — Write split-tagged JSON files (Req 16).
    12. **print_summary** — Print metrics to stdout (Req 16.8).

    The inference path contains **no** ``model_type``-keyed branches; all
    detector interaction is through the ``BaseDetector`` interface (Req 1.2,
    1.3). If a ``ConfigurationError`` is raised after the detector has been
    instantiated (e.g., during late validation against the model schema), the
    detector is released and its resources freed before terminating (Req 11.2).

    Args:
        config_path: Path to the YAML configuration file. When ``None``, an
            empty base configuration is used and only overrides apply.
        overrides: CLI/programmatic override dict deep-merged on top of the
            loaded configuration. Override values win.

    Returns:
        The assembled Evaluation_Report dict.

    Raises:
        ConfigurationError: If configuration validation fails (before or after
            detector instantiation).
        FileNotFoundError: If the checkpoint or dataset path does not exist.
        ModelNotFoundError: If the model type is not registered.
        RuntimeError: If checkpoint loading fails due to corruption, or if
            required metric fields are missing from the assembled report.

    Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 4.1, 10.3, 11.1, 11.2, 12.3
    """
    overrides = overrides or {}

    # -------------------------------------------------------------------------
    # Stage 1: Load and merge configuration (Req 3.1-3.5)
    # -------------------------------------------------------------------------
    config = load_and_merge_config(config_path, overrides)

    # -------------------------------------------------------------------------
    # Stage 2: Validate configuration before any I/O (Req 4.1, 11.1, 12.3)
    # -------------------------------------------------------------------------
    # Req 4.1: validate before instantiating Detector, loading checkpoints, or
    # loading Dataset. Req 11.1: terminate before Detector instantiated on error.
    validate_config(config)

    # -------------------------------------------------------------------------
    # Stage 3: Select device (Req 10.1, 10.2)
    # -------------------------------------------------------------------------
    # Req 10.2: log the selected device at INFO before loading the Detector.
    device = select_device()

    # -------------------------------------------------------------------------
    # Stage 4: Resolve checkpoint path (Req 5.1-5.4, 12.1)
    # -------------------------------------------------------------------------
    checkpoint_path = resolve_checkpoint(config)

    # -------------------------------------------------------------------------
    # Stage 5: Build detector via ModelRegistry (Req 1.1, 1.7, 13)
    # -------------------------------------------------------------------------
    _, model_type = _get_nested(config, ("model", "type"))
    _, model_config = _get_nested(config, ("model", "config"))
    model_config = model_config or {}

    detector = build_detector(model_type, model_config)

    # -------------------------------------------------------------------------
    # Stage 6: Load checkpoint + set_eval_mode + to_device (Req 1.4, 1.5, 1.6, 10.3, 11.2)
    # -------------------------------------------------------------------------
    # Req 11.2: if a ConfigurationError is raised after the Detector has been
    # instantiated, release the Detector and free resources before terminating.
    # We use a try/finally to ensure cleanup on any late error.
    try:
        # Req 1.4: invoke checkpoint loading exclusively through load_checkpoint.
        # Req 12.3: no inference can start while checkpoint error being raised.
        load_checkpoint_into(detector, checkpoint_path)

        # Req 1.5: invoke evaluation-mode activation exclusively through set_eval_mode.
        detector.set_eval_mode()

        # Req 1.6, 10.3: invoke device placement exclusively through to_device,
        # called exactly once with the selected device.
        detector.to_device(device)

        # -------------------------------------------------------------------------
        # Stage 7: Load split (Req 9.1, 9.2, 9.3, 14.1, 14.3)
        # -------------------------------------------------------------------------
        split_ds, class_names, idx_to_class, display_class_names = load_split(config)

        # Extract evaluation parameters with defaults.
        _, split = _get_nested(config, ("evaluation", "split"))
        found_conf, confidence_threshold = _get_nested(
            config, ("evaluation", "confidence_threshold")
        )
        if not found_conf or confidence_threshold is None:
            confidence_threshold = 0.25
        found_iou, iou_threshold = _get_nested(
            config, ("evaluation", "iou_threshold")
        )
        if not found_iou or iou_threshold is None:
            iou_threshold = 0.5
        found_input_size, input_size = _get_nested(
            config, ("model", "config", "input_size")
        )
        if not found_input_size or input_size is None:
            input_size = 640
        _, output_dir = _get_nested(config, ("evaluation", "output_dir"))
        _, dataset_path = _get_nested(config, ("dataset", "path"))

        # -------------------------------------------------------------------------
        # Stage 8: Run inference (Req 1.2, 1.3, 7, 10.4)
        # -------------------------------------------------------------------------
        # Req 1.2: invoke inference exclusively through BaseDetector.forward().
        # Req 1.3: no conditional branches keyed on model type in inference path.
        # Sync the evaluation confidence threshold to the detector so that its
        # internal filtering (if any) uses the evaluation-level threshold rather
        # than the model-config default.  This is done via a generic attribute
        # set (no model_type-keyed branch) and is a no-op for detectors that do
        # not perform internal confidence filtering.
        if hasattr(detector, "confidence_threshold"):
            detector.confidence_threshold = float(confidence_threshold)
        predictions, ground_truths, errors = run_inference(
            detector=detector,
            split_ds=split_ds,
            device=device,
            input_size=int(input_size),
            idx_to_class=idx_to_class,
        )

    except ConfigurationError:
        # Req 11.2: release detector and free resources on late ConfigurationError.
        _release_detector(detector)
        raise
    except Exception:
        # For any other exception after detector instantiation, also clean up.
        _release_detector(detector)
        raise

    # -------------------------------------------------------------------------
    # Stage 9: Compute metrics (Req 8.1-8.5)
    # -------------------------------------------------------------------------
    logger.info("Computing metrics...")
    metrics = compute_all_metrics(
        predictions=predictions,
        ground_truths=ground_truths,
        class_names=class_names,
        confidence_threshold=float(confidence_threshold),
        iou_threshold=float(iou_threshold),
        errors=errors,
    )

    # -------------------------------------------------------------------------
    # Stage 10: Assemble report (Req 8.6, 8.7, 9.4, 16.2, 17.4)
    # -------------------------------------------------------------------------
    # Convert confusion matrix to list for JSON serialization.
    confusion_matrix = metrics["confusion_matrix"]
    if hasattr(confusion_matrix, "tolist"):
        confusion_matrix = confusion_matrix.tolist()

    report = assemble_report(
        checkpoint_path=str(checkpoint_path),
        model_type=model_type,
        model_config=model_config,
        dataset_path=str(dataset_path),
        split=split,
        num_images=len(predictions),
        num_classes=len(class_names),
        class_names=class_names,
        confidence_threshold=float(confidence_threshold),
        iou_threshold=float(iou_threshold),
        metrics=metrics,
        confusion_matrix=confusion_matrix,
        errors=errors,
        display_class_names=display_class_names,
    )

    # -------------------------------------------------------------------------
    # Stage 11: Write outputs (Req 9.5, 16.1, 16.3-16.7)
    # -------------------------------------------------------------------------
    write_outputs(
        report=report,
        predictions=predictions,
        ground_truths=ground_truths,
        output_dir=output_dir,
        split=split,
        checkpoint_path=checkpoint_path,
    )

    # -------------------------------------------------------------------------
    # Stage 12: Print summary (Req 16.8)
    # -------------------------------------------------------------------------
    print_summary(
        model_type=model_type,
        split=split,
        num_images=len(predictions),
        metrics=report["metrics"],
    )

    # Release detector resources after successful completion.
    _release_detector(detector)

    return report


def _release_detector(detector) -> None:
    """Release detector resources to free memory.

    Attempts to delete the detector's underlying model and clear CUDA cache
    when available. This is called on both successful completion and on late
    errors after detector instantiation (Req 11.2).

    Args:
        detector: The detector instance to release.
    """
    try:
        # Try to delete the underlying model to free GPU memory.
        if hasattr(detector, "_model"):
            del detector._model
        elif hasattr(detector, "model"):
            del detector.model
        # Clear CUDA cache if available.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # Ignore cleanup errors; we're already handling another exception.
        pass


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser, retaining every prior argument.

    Creates an :class:`argparse.ArgumentParser` that accepts all arguments
    supported by the evaluation script prior to the generic-evaluation
    enhancement. Each argument maps into a structured config section
    (``model``, ``dataset``, ``evaluation``, ``checkpoint``) as a CLI override
    that is deep-merged on top of the loaded YAML configuration.

    Argument-to-override mapping:

    +-----------------+----------------------------------------------+
    | Argument        | Override path                                 |
    +=================+==============================================+
    | --config        | (config file path — not an override)         |
    | --checkpoint    | checkpoint.path                              |
    | --run-id        | checkpoint.run_id                            |
    | --checkpoint-dir| checkpoint.checkpoint_dir                    |
    | --model-type    | model.type (deprecated when --config         |
    |                 | supplies it → WARNING)                       |
    | --input-size    | evaluation.input_size / model.config.input_size|
    | --num-classes   | model.config.num_classes                     |
    | --dataset       | dataset.path                                 |
    | --split         | evaluation.split                             |
    | --val-split     | evaluation.val_split                         |
    | --confidence    | evaluation.confidence_threshold              |
    | --iou           | evaluation.iou_threshold                     |
    | --output-dir    | evaluation.output_dir                        |
    | --verbose / -v  | logging level                                |
    +-----------------+----------------------------------------------+

    Returns:
        The configured :class:`argparse.ArgumentParser`.

    Requirements: 17.1
    """
    parser = argparse.ArgumentParser(
        description="Evaluate detection models (YOLO26, SSD MobileNetV3, etc.) on RDD2022"
    )

    # Primary arguments (recommended usage)
    parser.add_argument(
        "--config",
        type=str,
        help="Path to training config YAML file (e.g., model/configs/train_yolo26.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Direct path to .pt checkpoint file",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="UUID of the training run to evaluate (looks for checkpoint in checkpoint-dir)",
    )

    # Model configuration (for legacy usage without config file)
    parser.add_argument(
        "--model-type",
        type=str,
        help="Model type from ModelRegistry (e.g., 'yolo26', 'ssd_mobilenet', 'yolov6'). "
        "Deprecated when --config supplies model.type.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        help="Model input resolution (overrides config)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        help="Number of classes (overrides config)",
    )

    # Dataset and evaluation settings
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to RDD2022 dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        help="Which split to evaluate: train, val, or test",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        help="Validation split ratio (for recreating train/val split)",
    )

    # Evaluation thresholds
    parser.add_argument(
        "--confidence",
        type=float,
        help="Confidence threshold for predictions (overrides config)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        help="IoU threshold for matching (overrides config)",
    )

    # Output settings
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Base directory for checkpoints (used with --run-id)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save evaluation report and predictions",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    return parser


def main() -> None:
    """Parse CLI arguments, map them into config overrides, and run evaluation.

    Parses arguments via :func:`build_arg_parser`, sets up logging based on
    ``--verbose``, maps each CLI argument into the appropriate structured config
    section (``model``, ``dataset``, ``evaluation``, ``checkpoint``) as an
    override, and calls :func:`evaluate` with the config path and overrides.

    Deprecated arguments are identified and a WARNING is logged naming the
    deprecated argument and its replacement before execution continues
    (Req 17.2). Currently deprecated:

    * ``--model-type`` when ``--config`` is also supplied (the config file's
      ``model.type`` is the preferred source; use ``--config`` instead).

    Requirements: 17.1, 17.2
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    # Set up logging based on verbosity.
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Req 17.2: warn about deprecated arguments and their replacements.
    if args.model_type and args.config:
        logger.warning(
            "--model-type is deprecated when --config is supplied. "
            "Use the 'model.type' field in the config file instead. "
            "The CLI value will be used as an override for this run."
        )

    # Build the overrides dict from CLI arguments.
    overrides: dict = {}

    # Checkpoint section: path XOR run_id.
    if args.checkpoint:
        overrides.setdefault("checkpoint", {})["path"] = args.checkpoint
    if args.run_id:
        overrides.setdefault("checkpoint", {})["run_id"] = args.run_id
    if args.checkpoint_dir != "./checkpoints":
        overrides.setdefault("checkpoint", {})["checkpoint_dir"] = args.checkpoint_dir

    # Model section.
    if args.model_type:
        overrides.setdefault("model", {})["type"] = args.model_type
    if args.input_size is not None:
        overrides.setdefault("model", {}).setdefault("config", {})["input_size"] = (
            args.input_size
        )
        overrides.setdefault("evaluation", {})["input_size"] = args.input_size
    if args.num_classes is not None:
        overrides.setdefault("model", {}).setdefault("config", {})["num_classes"] = (
            args.num_classes
        )

    # Dataset section.
    if args.dataset:
        overrides.setdefault("dataset", {})["path"] = args.dataset

    # Evaluation section.
    if args.split:
        overrides.setdefault("evaluation", {})["split"] = args.split
    if args.val_split is not None:
        overrides.setdefault("evaluation", {})["val_split"] = args.val_split
    if args.confidence is not None:
        overrides.setdefault("evaluation", {})["confidence_threshold"] = args.confidence
    if args.iou is not None:
        overrides.setdefault("evaluation", {})["iou_threshold"] = args.iou
    if args.output_dir:
        overrides.setdefault("evaluation", {})["output_dir"] = args.output_dir

    # Call the evaluate() function with config path and overrides.
    evaluate(config_path=args.config, overrides=overrides)


if __name__ == "__main__":
    main()
