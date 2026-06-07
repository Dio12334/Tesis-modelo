"""Empirical measurement of mosaic emptiness over a full train epoch.

Iterates every annotation in the RDD2022 train split, calls
``RDD2022TorchDataset._build_mosaic`` on it (using the production sampling
logic), and records whether the resulting mosaic carried any bboxes plus
the upstream causes (primary empty? companions empty? centre-crop
discarded all bboxes?).

This tool is read-only with respect to the dataset and the model. It writes
nothing except a stdout report and (optionally) a JSON summary file.

Usage:
    python -m model.tools.measure_mosaic_emptiness
    python -m model.tools.measure_mosaic_emptiness --root model/data/rdd2022/complete --input-size 640
    python -m model.tools.measure_mosaic_emptiness --json-out checkpoints/mosaic_stats.json

Notes:
- The tool seeds ``random`` and ``numpy.random`` deterministically so the
  measured numbers are reproducible across invocations.
- One epoch == one mosaic build per ``len(dataset)`` annotations. With
  38,385 annotations and ~10 ms/build, expect ~6-8 minutes wall-clock on
  WSL with a warm filesystem cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from model.datasets.rdd2022 import RDD2022Dataset
from model.training.train_detection import RDD2022TorchDataset

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path("model/data/rdd2022/complete")
DEFAULT_INPUT_SIZE = 640
DEFAULT_SEED = 42


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap number of mosaics built (default: full epoch).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=2000,
        help="Print a progress line every N mosaics.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.root.exists():
        print(f"ERROR: dataset root not found: {args.root}", file=sys.stderr)
        return 2

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading RDD2022 train split from {args.root}...")
    base = RDD2022Dataset(subset="train")
    base.load(args.root)
    annotations = base.get_annotations()
    n = len(annotations)
    if n == 0:
        print("ERROR: no annotations loaded", file=sys.stderr)
        return 2

    n_empty_in_dataset = sum(1 for a in annotations if not a.bounding_boxes)
    n_nonempty_in_dataset = n - n_empty_in_dataset
    print(
        f"Dataset: {n} annotations ({n_nonempty_in_dataset} non-empty, "
        f"{n_empty_in_dataset} empty, "
        f"{100.0 * n_empty_in_dataset / n:.1f}% empty)"
    )

    ds = RDD2022TorchDataset(
        dataset=base,
        input_size=args.input_size,
        augmentation=None,
        mosaic=1.0,
        mixup=0.0,
    )

    # Whether RDD2022TorchDataset filters companions to non-empty pool.
    # If so, captured randint(0, pool_n - 1) values are indices INTO the
    # non-empty pool, not annotation indices directly.
    nonempty_pool = getattr(ds, "_nonempty_indices", None)
    pool_n = len(nonempty_pool) if nonempty_pool is not None else n
    companion_bound_lo, companion_bound_hi = 0, pool_n - 1
    print(
        f"Companion sampling pool: {pool_n} indices "
        f"({'filtered to non-empty' if nonempty_pool is not None and pool_n != n else 'unfiltered'})"
    )

    total = n if args.max_samples is None else min(n, args.max_samples)
    print(f"Building {total} mosaics at {args.input_size}x{args.input_size}...")

    # Counters
    empty_mosaics = 0  # 0 bboxes after final centre-crop
    primary_empty_count = 0
    companions_empty_histogram = Counter()  # k -> #mosaics where k of 3 companions empty
    bbox_count_histogram = Counter()  # k -> #mosaics with k final bboxes
    primary_empty_and_mosaic_empty = 0
    primary_nonempty_and_mosaic_empty = 0  # the worst case: had bboxes, all dropped
    bboxes_seen_pre_crop = 0  # not directly measurable without re-instrumenting;
    # we approximate by counting bbox-bearing companions

    t0 = time.time()
    last_print = t0

    # We DON'T re-seed inside the loop, so mosaic randomness is realistic
    # and independent across iterations.
    for i in range(total):
        ann = annotations[i]
        primary_empty = not ann.bounding_boxes
        if primary_empty:
            primary_empty_count += 1

        # Inspect companion sampling decisions BEFORE calling _build_mosaic.
        # We can't peek into _build_mosaic without instrumenting, so instead
        # we replicate its companion sampling using the same RNG state.
        # However that consumes RNG draws that _build_mosaic also makes,
        # which would corrupt its randomness. So instead we measure
        # companions AFTER the fact by checking the # of bboxes that
        # *could have* contributed (from primary + 3 random companions).
        #
        # For an unbiased estimate, we fork the RNG: sample 3 indices with
        # a copy of the state, count their non-emptiness, then restore
        # state. This keeps _build_mosaic's behaviour identical while
        # giving us the same companion identities (because the first 3
        # random.randint calls inside _build_mosaic come right after our
        # peek -- but wait, _build_mosaic also calls random.uniform twice
        # for xc, yc BEFORE the companion draws, so peeking with
        # random.randint won't predict the actual companions).
        #
        # Simpler: just call _build_mosaic, then re-seed and replicate
        # _build_mosaic's RNG sequence to figure out which companions were
        # used. To avoid that complexity, we instead patch random.randint
        # transiently to capture the companion indices.
        captured_companions: List[int] = []
        real_randint = random.randint

        def _capturing_randint(a, b):
            v = real_randint(a, b)
            # _build_mosaic samples companions via random.randint over the
            # non-empty pool: randint(0, pool_n - 1). Translate through the
            # pool to recover the actual annotation index for emptiness check.
            if a == companion_bound_lo and b == companion_bound_hi:
                if nonempty_pool is not None and pool_n != n:
                    captured_companions.append(nonempty_pool[v])
                else:
                    captured_companions.append(v)
            return v

        random.randint = _capturing_randint
        try:
            _, bboxes = ds._build_mosaic(i)
        finally:
            random.randint = real_randint

        n_companions_empty = sum(
            1 for c in captured_companions if not annotations[c].bounding_boxes
        )
        # Should always be 3 captured; defensive check.
        if len(captured_companions) != 3:
            print(
                f"WARN: captured {len(captured_companions)} companions at i={i} "
                f"(expected 3); skipping", file=sys.stderr,
            )
            continue
        companions_empty_histogram[n_companions_empty] += 1

        nb = len(bboxes)
        bbox_count_histogram[nb] += 1
        if nb == 0:
            empty_mosaics += 1
            if primary_empty:
                primary_empty_and_mosaic_empty += 1
            else:
                primary_nonempty_and_mosaic_empty += 1
            # If at least one of (primary, companions) had bboxes but the
            # mosaic ended up empty, that's evidence of crop-induced loss.
            primary_had = not primary_empty
            companions_had = (3 - n_companions_empty) > 0
            if primary_had or companions_had:
                bboxes_seen_pre_crop += 1

        if (i + 1) % args.progress_every == 0:
            now = time.time()
            elapsed = now - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(
                f"  [{i + 1:>6}/{total}] "
                f"empty_mosaics={empty_mosaics} "
                f"({100.0 * empty_mosaics / (i + 1):.1f}%)  "
                f"rate={rate:.1f} mosaics/s  eta={eta:.0f}s"
            )
            last_print = now

    elapsed = time.time() - t0

    # ------------------------ Report ------------------------
    print()
    print("=" * 72)
    print("MOSAIC EMPTINESS REPORT")
    print("=" * 72)
    print(f"Total mosaics built     : {total}")
    print(f"Wall-clock              : {elapsed:.1f}s ({total / elapsed:.1f} mosaics/s)")
    print()
    print("--- Empty-mosaic breakdown ---")
    print(
        f"Mosaics with 0 bboxes   : {empty_mosaics} "
        f"({100.0 * empty_mosaics / total:.2f}%)"
    )
    print(
        f"  primary was empty     : {primary_empty_and_mosaic_empty} "
        f"({100.0 * primary_empty_and_mosaic_empty / total:.2f}% of all)"
    )
    print(
        f"  primary had bboxes    : {primary_nonempty_and_mosaic_empty} "
        f"({100.0 * primary_nonempty_and_mosaic_empty / total:.2f}% of all)"
    )
    print(
        f"  -> bboxes lost to     : {bboxes_seen_pre_crop} "
        f"({100.0 * bboxes_seen_pre_crop / total:.2f}% of all)"
    )
    print(
        f"     centre-crop alone"
    )
    print()
    print("--- Companion-emptiness distribution ---")
    print("(How many of the 3 random companions were empty annotations)")
    for k in sorted(companions_empty_histogram):
        cnt = companions_empty_histogram[k]
        print(f"  {k}/3 companions empty  : {cnt:>6} ({100.0 * cnt / total:.2f}%)")
    expected_p_empty = n_empty_in_dataset / n
    expected_dist = [
        (1 - expected_p_empty) ** 3,
        3 * expected_p_empty * (1 - expected_p_empty) ** 2,
        3 * expected_p_empty ** 2 * (1 - expected_p_empty),
        expected_p_empty ** 3,
    ]
    print(
        f"  (binomial expectation : "
        f"{[round(p, 3) for p in expected_dist]})"
    )
    print()
    print("--- Bbox-count distribution per mosaic ---")
    cum_at_zero = 0
    for k in sorted(bbox_count_histogram):
        cnt = bbox_count_histogram[k]
        print(f"  {k:>3} bboxes            : {cnt:>6} ({100.0 * cnt / total:.2f}%)")
        if k == 0:
            cum_at_zero = cnt
        if k >= 12:
            break  # truncate long tail
    bbox_total = sum(k * v for k, v in bbox_count_histogram.items())
    print(f"  Total bboxes emitted  : {bbox_total} "
          f"(avg {bbox_total / total:.2f} per mosaic)")
    print()
    print("--- Dataset baseline ---")
    print(
        f"  empty annotations     : {n_empty_in_dataset}/{n} "
        f"({100.0 * n_empty_in_dataset / n:.2f}%)"
    )
    print(
        f"  expected P(all 4 sources empty) "
        f"= {(n_empty_in_dataset / n) ** 4:.4f} "
        f"(naive independence baseline)"
    )
    print()

    summary: Dict[str, Any] = {
        "input_size": args.input_size,
        "seed": args.seed,
        "total_mosaics": total,
        "wall_clock_s": elapsed,
        "empty_mosaics": empty_mosaics,
        "empty_mosaics_pct": 100.0 * empty_mosaics / total,
        "primary_empty_and_mosaic_empty": primary_empty_and_mosaic_empty,
        "primary_nonempty_and_mosaic_empty": primary_nonempty_and_mosaic_empty,
        "centre_crop_only_loss": bboxes_seen_pre_crop,
        "companions_empty_histogram": dict(companions_empty_histogram),
        "bbox_count_histogram": dict(bbox_count_histogram),
        "dataset_total": n,
        "dataset_empty": n_empty_in_dataset,
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"Wrote summary to {args.json_out.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
