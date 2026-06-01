"""Property-based tests for split-tagged output filename injectivity.

Feature: generic-evaluation-script
Property 17: Split-tagged output filenames are injective

For any two distinct split values from ``{train, val, test}``, the generated
report filenames are distinct and the generated predictions filenames are
distinct (injective mapping from split to filename).

The output file naming pattern (from the design document) is:
- Report: ``{split}_evaluation_report.json``
- Predictions: ``{split}_inference.json``

This guarantees that train/val/test runs never overwrite each other.

**Validates: Requirements 9.5, 16.6**
"""

from itertools import combinations, permutations

from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Filename generation functions (matching the design specification)
# ---------------------------------------------------------------------------


def get_report_filename(split: str) -> str:
    """Generate the report filename for a given split.

    The pattern is ``{split}_evaluation_report.json`` as specified in the
    design document's "Output file naming" section.

    Args:
        split: One of ``train``, ``val``, or ``test``.

    Returns:
        The report filename for the given split.
    """
    return f"{split}_evaluation_report.json"


def get_predictions_filename(split: str) -> str:
    """Generate the predictions filename for a given split.

    The pattern is ``{split}_inference.json`` as specified in the design
    document's "Output file naming" section.

    Args:
        split: One of ``train``, ``val``, or ``test``.

    Returns:
        The predictions filename for the given split.
    """
    return f"{split}_inference.json"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_SPLITS = ["train", "val", "test"]


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Strategy for a single valid split value.
_SPLIT = st.sampled_from(VALID_SPLITS)

# Strategy for two distinct split values (ordered pair).
_DISTINCT_SPLIT_PAIR = st.sampled_from(
    [(a, b) for a, b in permutations(VALID_SPLITS, 2)]
)


# ---------------------------------------------------------------------------
# Property 17
# ---------------------------------------------------------------------------


class TestProperty17SplitFilenamesInjective:
    """Property 17: Split-tagged output filenames are injective.

    For any two distinct split values from ``{train, val, test}``, the generated
    report filenames are distinct and the generated predictions filenames are
    distinct.

    **Validates: Requirements 9.5, 16.6**
    """

    @given(split_pair=_DISTINCT_SPLIT_PAIR)
    @settings(max_examples=100)
    def test_distinct_splits_produce_distinct_report_filenames(self, split_pair):
        # Feature: generic-evaluation-script, Property 17: Split filenames injective
        """Distinct splits produce distinct report filenames.

        For any two distinct split values ``split_a`` and ``split_b``, the
        report filenames ``get_report_filename(split_a)`` and
        ``get_report_filename(split_b)`` are distinct.

        **Validates: Requirements 9.5, 16.6**
        """
        split_a, split_b = split_pair

        # Precondition: splits are distinct.
        assert split_a != split_b

        report_a = get_report_filename(split_a)
        report_b = get_report_filename(split_b)

        # Injectivity: distinct inputs produce distinct outputs.
        assert report_a != report_b, (
            f"Report filenames should be distinct for distinct splits, "
            f"but got '{report_a}' for both '{split_a}' and '{split_b}'"
        )

    @given(split_pair=_DISTINCT_SPLIT_PAIR)
    @settings(max_examples=100)
    def test_distinct_splits_produce_distinct_predictions_filenames(self, split_pair):
        # Feature: generic-evaluation-script, Property 17: Split filenames injective
        """Distinct splits produce distinct predictions filenames.

        For any two distinct split values ``split_a`` and ``split_b``, the
        predictions filenames ``get_predictions_filename(split_a)`` and
        ``get_predictions_filename(split_b)`` are distinct.

        **Validates: Requirements 9.5, 16.6**
        """
        split_a, split_b = split_pair

        # Precondition: splits are distinct.
        assert split_a != split_b

        predictions_a = get_predictions_filename(split_a)
        predictions_b = get_predictions_filename(split_b)

        # Injectivity: distinct inputs produce distinct outputs.
        assert predictions_a != predictions_b, (
            f"Predictions filenames should be distinct for distinct splits, "
            f"but got '{predictions_a}' for both '{split_a}' and '{split_b}'"
        )

    @given(split=_SPLIT)
    @settings(max_examples=100)
    def test_report_and_predictions_filenames_are_distinct(self, split):
        # Feature: generic-evaluation-script, Property 17: Split filenames injective
        """Report and predictions filenames are distinct for the same split.

        For any split value, the report filename and predictions filename are
        distinct (they use different suffixes).

        **Validates: Requirements 9.5, 16.6**
        """
        report = get_report_filename(split)
        predictions = get_predictions_filename(split)

        assert report != predictions, (
            f"Report and predictions filenames should be distinct for split "
            f"'{split}', but both are '{report}'"
        )

    @given(split=_SPLIT)
    @settings(max_examples=100)
    def test_filenames_contain_split_value(self, split):
        # Feature: generic-evaluation-script, Property 17: Split filenames injective
        """Filenames contain the split value as a prefix.

        The split value is incorporated into the filename as a prefix, ensuring
        that the filename clearly identifies which split it belongs to.

        **Validates: Requirements 9.5, 16.6**
        """
        report = get_report_filename(split)
        predictions = get_predictions_filename(split)

        assert report.startswith(f"{split}_"), (
            f"Report filename '{report}' should start with '{split}_'"
        )
        assert predictions.startswith(f"{split}_"), (
            f"Predictions filename '{predictions}' should start with '{split}_'"
        )


# ---------------------------------------------------------------------------
# Exhaustive example-based tests
# ---------------------------------------------------------------------------


class TestSplitFilenamesExhaustive:
    """Exhaustive example-based tests for split-tagged filenames.

    These tests verify the exact filename patterns specified in the design
    document and confirm injectivity across all valid split combinations.

    **Validates: Requirements 9.5, 16.6**
    """

    def test_all_report_filenames_are_distinct(self):
        """All three report filenames are pairwise distinct.

        **Validates: Requirements 9.5, 16.6**
        """
        filenames = [get_report_filename(split) for split in VALID_SPLITS]

        # All filenames are distinct (no duplicates).
        assert len(filenames) == len(set(filenames)), (
            f"Report filenames should all be distinct, but got: {filenames}"
        )

    def test_all_predictions_filenames_are_distinct(self):
        """All three predictions filenames are pairwise distinct.

        **Validates: Requirements 9.5, 16.6**
        """
        filenames = [get_predictions_filename(split) for split in VALID_SPLITS]

        # All filenames are distinct (no duplicates).
        assert len(filenames) == len(set(filenames)), (
            f"Predictions filenames should all be distinct, but got: {filenames}"
        )

    def test_exact_report_filename_patterns(self):
        """Report filenames match the exact patterns from the design document.

        | split | report filename |
        |-------|-----------------|
        | train | train_evaluation_report.json |
        | val   | val_evaluation_report.json |
        | test  | test_evaluation_report.json |

        **Validates: Requirements 9.5, 16.6**
        """
        assert get_report_filename("train") == "train_evaluation_report.json"
        assert get_report_filename("val") == "val_evaluation_report.json"
        assert get_report_filename("test") == "test_evaluation_report.json"

    def test_exact_predictions_filename_patterns(self):
        """Predictions filenames match the exact patterns from the design document.

        | split | predictions filename |
        |-------|----------------------|
        | train | train_inference.json |
        | val   | val_inference.json |
        | test  | test_inference.json |

        **Validates: Requirements 9.5, 16.6**
        """
        assert get_predictions_filename("train") == "train_inference.json"
        assert get_predictions_filename("val") == "val_inference.json"
        assert get_predictions_filename("test") == "test_inference.json"

    def test_no_filename_collisions_across_all_pairs(self):
        """No filename collisions exist across all split pairs.

        For every pair of distinct splits, both report and predictions filenames
        are distinct.

        **Validates: Requirements 9.5, 16.6**
        """
        for split_a, split_b in combinations(VALID_SPLITS, 2):
            # Report filenames are distinct.
            report_a = get_report_filename(split_a)
            report_b = get_report_filename(split_b)
            assert report_a != report_b, (
                f"Report collision: '{split_a}' and '{split_b}' both produce "
                f"'{report_a}'"
            )

            # Predictions filenames are distinct.
            predictions_a = get_predictions_filename(split_a)
            predictions_b = get_predictions_filename(split_b)
            assert predictions_a != predictions_b, (
                f"Predictions collision: '{split_a}' and '{split_b}' both "
                f"produce '{predictions_a}'"
            )

    def test_report_and_predictions_never_collide(self):
        """Report and predictions filenames never collide across any splits.

        No report filename equals any predictions filename, even across
        different splits.

        **Validates: Requirements 9.5, 16.6**
        """
        report_filenames = {get_report_filename(s) for s in VALID_SPLITS}
        predictions_filenames = {get_predictions_filename(s) for s in VALID_SPLITS}

        # No overlap between report and predictions filename sets.
        overlap = report_filenames & predictions_filenames
        assert not overlap, (
            f"Report and predictions filenames should not overlap, "
            f"but found: {overlap}"
        )
