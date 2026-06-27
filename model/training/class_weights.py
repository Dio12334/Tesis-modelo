"""Class re-weighting for imbalanced detection — SR-WBCE.

Implements the **Smoothed Root-Weighted Binary Cross-Entropy** weighting from
YOLO-RD (Wang et al., Neurocomputing 662 (2026) 132000), Eq. 17:

    w_k = max_j( N_j^(1/n) ) / N_k^(1/n)

where ``N_k`` is the number of training instances of class ``k`` and ``n`` is a
smoothing exponent (the paper's best value is ``n = 4``). The most frequent class
gets weight ≈ 1.0; rarer classes are up-weighted, with the n-th root *smoothing*
the spread (larger ``n`` → weights closer to 1.0) to avoid the training
instability of plain inverse-frequency weighting.

The resulting vector is consumed by Ultralytics' ``v8DetectionLoss`` /
``E2EDetectLoss`` via ``model.class_weights`` — they already multiply it into the
classification BCE, so no loss subclass is needed.
"""

from typing import List, Sequence

__all__ = ["sr_wbce_weights", "count_class_instances"]


def sr_wbce_weights(counts: Sequence[int], n: float = 4.0) -> List[float]:
    """Compute SR-WBCE per-class weights from instance counts.

    Args:
        counts: Per-class training-instance counts, indexed by class id.
        n: Smoothing exponent (n-th root). The paper's best is 4.

    Returns:
        A list of per-class weights, same length as ``counts``. The most frequent
        class is ≈ 1.0; rarer classes are > 1.0.

    Notes:
        Counts are clamped to a minimum of 1 to avoid division by zero (a class
        absent from training is treated as the rarest; it contributes no loss
        anyway). Returns an empty list for empty input.
    """
    if n <= 0:
        raise ValueError(f"SR-WBCE exponent n must be > 0, got {n}")
    if len(counts) == 0:
        return []
    roots = [max(int(c), 1) ** (1.0 / n) for c in counts]
    max_root = max(roots)
    return [max_root / r for r in roots]


def count_class_instances(dataset, class_names: Sequence[str]) -> List[int]:
    """Count bounding-box instances per class over a detection dataset.

    Iterates the dataset's annotations (no image decoding) and tallies each
    ``bbox.class_label`` against ``class_names``. Labels not in ``class_names``
    are ignored.

    Args:
        dataset: A ``BaseDataset`` (or any iterable of annotations exposing
            ``bounding_boxes`` with a ``class_label`` attribute).
        class_names: Class names in the model's index order (typically
            ``dataset.get_class_names()``, i.e. sorted unique labels).

    Returns:
        Per-class instance counts, indexed to match ``class_names``.
    """
    idx = {name: i for i, name in enumerate(class_names)}
    counts = [0] * len(class_names)
    for annotation in dataset:
        for bbox in annotation.bounding_boxes:
            i = idx.get(bbox.class_label)
            if i is not None:
                counts[i] += 1
    return counts
