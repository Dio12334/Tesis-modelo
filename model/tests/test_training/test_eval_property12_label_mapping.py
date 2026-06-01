"""Property-based tests for label-index to class-name mapping.

Feature: generic-evaluation-script
Property 12: Label index to class-name mapping

For any integer label index and any index-to-class mapping, ``map_label``
returns the class name when the index is present in the mapping and the literal
``class_<index>`` otherwise.

These tests exercise the real ``map_label`` function in
``model/training/evaluate_detection.py``, whose contract (Req 7.4) is::

    map_label(index, idx_to_class) == idx_to_class.get(index, f"class_{index}")

The strategies below build arbitrary integer indices and arbitrary
``dict[int, str]`` index-to-class mappings, deliberately drawing the queried
index *both* from inside the mapping's key set (to exercise the "present"
branch) and from arbitrary integers (to exercise the "absent" branch). This
ensures the bidirectional (iff) property is covered in both directions.

**Validates: Requirements 7.4**
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from model.training.evaluate_detection import map_label


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Arbitrary integer label indices, including negatives and large values, so the
# fallback string formatting is exercised across the full integer space.
_INDEXES = st.integers(min_value=-1000, max_value=1000)

# Class-name strings. Non-empty, drawn from a realistic alphabet plus arbitrary
# text, so a present mapping returns a genuine, identifiable class name (and the
# class name is never accidentally equal to a "class_<index>" fallback unless
# explicitly constructed that way).
_CLASS_NAMES = st.one_of(
    st.sampled_from(["crack", "pothole", "longitudinal", "alligator", "repair"]),
    st.text(min_size=1, max_size=12),
)

# Arbitrary index-to-class mappings: dict[int, str]. Empty maps are allowed so
# the "absent" branch is always reachable.
_IDX_TO_CLASS = st.dictionaries(keys=_INDEXES, values=_CLASS_NAMES, max_size=10)


class TestProperty12LabelMapping:
    """Property 12: Label index to class-name mapping.

    **Validates: Requirements 7.4**
    """

    @given(index=_INDEXES, idx_to_class=_IDX_TO_CLASS)
    @settings(max_examples=100)
    def test_mapping_is_present_class_name_else_fallback(self, index, idx_to_class):
        # Feature: generic-evaluation-script, Property 12: Label mapping
        """Mapped label is the class name iff present, else ``class_<index>``.

        **Validates: Requirements 7.4**
        """
        result = map_label(index, idx_to_class)

        if index in idx_to_class:
            # Present index -> the registered class name, exactly.
            assert result == idx_to_class[index]
        else:
            # Absent index -> the literal class_<index> fallback.
            assert result == f"class_{index}"

    @given(
        idx_to_class=st.dictionaries(
            keys=_INDEXES, values=_CLASS_NAMES, min_size=1, max_size=10
        ),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_present_index_returns_registered_name(self, idx_to_class, data):
        # Feature: generic-evaluation-script, Property 12: Label mapping
        """Drawing an index from the mapping always returns its class name.

        Exercises the "present" branch directly by sampling a key that is
        guaranteed to exist in the mapping.

        **Validates: Requirements 7.4**
        """
        index = data.draw(st.sampled_from(sorted(idx_to_class.keys())))

        assert map_label(index, idx_to_class) == idx_to_class[index]

    @given(index=_INDEXES, idx_to_class=_IDX_TO_CLASS)
    @settings(max_examples=100)
    def test_absent_index_returns_class_underscore_index(self, index, idx_to_class):
        # Feature: generic-evaluation-script, Property 12: Label mapping
        """Removing an index from the mapping forces the ``class_<index>`` fallback.

        Exercises the "absent" branch directly by ensuring the queried index is
        not a key of the mapping.

        **Validates: Requirements 7.4**
        """
        # Guarantee the index is absent regardless of what was generated.
        pruned = {k: v for k, v in idx_to_class.items() if k != index}

        assert map_label(index, pruned) == f"class_{index}"

    def test_empty_mapping_always_falls_back(self):
        # Feature: generic-evaluation-script, Property 12: Label mapping
        """With an empty mapping, every index maps to ``class_<index>``.

        **Validates: Requirements 7.4**
        """
        for index in (-5, -1, 0, 1, 2, 42, 999):
            assert map_label(index, {}) == f"class_{index}"
