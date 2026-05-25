"""Unit tests for TargetMapper."""

import tempfile
from pathlib import Path

import pytest
import yaml

from model.datasets.target_mapper import TargetMapper
from model.exceptions import ConfigurationError, UnmappedClassError


@pytest.fixture
def sample_config(tmp_path):
    """Create a sample mapping config YAML file."""
    config = {
        "taxonomy": [
            "bache",
            "fisura_longitudinal",
            "fisura_transversal",
            "piel_de_cocodrilo",
        ],
        "mappings": {
            "D00": "fisura_longitudinal",
            "D10": "fisura_transversal",
            "D20": "piel_de_cocodrilo",
            "D40": "bache",
        },
        "default_class": None,
    }
    config_path = tmp_path / "mapping.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
def config_with_default(tmp_path):
    """Create a mapping config with a default class."""
    config = {
        "taxonomy": ["bache", "fisura_longitudinal", "otro"],
        "mappings": {
            "D00": "fisura_longitudinal",
            "D40": "bache",
        },
        "default_class": "otro",
    }
    config_path = tmp_path / "mapping_default.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


class TestTargetMapperInit:
    def test_loads_config(self, sample_config):
        mapper = TargetMapper(sample_config)
        assert len(mapper.mappings) == 4
        assert len(mapper.taxonomy) == 4
        assert mapper.strict is True
        assert mapper.default_class is None

    def test_strict_flag(self, sample_config):
        mapper = TargetMapper(sample_config, strict=False)
        assert mapper.strict is False


class TestMapClass:
    def test_maps_known_class(self, sample_config):
        mapper = TargetMapper(sample_config)
        assert mapper.map_class("D00") == "fisura_longitudinal"
        assert mapper.map_class("D10") == "fisura_transversal"
        assert mapper.map_class("D20") == "piel_de_cocodrilo"
        assert mapper.map_class("D40") == "bache"

    def test_strict_raises_for_unknown(self, sample_config):
        mapper = TargetMapper(sample_config, strict=True)
        with pytest.raises(UnmappedClassError) as exc_info:
            mapper.map_class("D99")
        assert exc_info.value.source_class == "D99"
        assert "D00" in exc_info.value.available_classes

    def test_non_strict_with_default(self, config_with_default):
        mapper = TargetMapper(config_with_default, strict=False)
        assert mapper.map_class("UNKNOWN") == "otro"

    def test_non_strict_no_default_raises(self, sample_config):
        mapper = TargetMapper(sample_config, strict=False)
        with pytest.raises(UnmappedClassError):
            mapper.map_class("UNKNOWN")


class TestReverseMap:
    def test_reverse_single(self, sample_config):
        mapper = TargetMapper(sample_config)
        result = mapper.reverse_map("bache")
        assert result == ["D40"]

    def test_reverse_many_to_one(self, tmp_path):
        config = {
            "taxonomy": ["crack"],
            "mappings": {
                "D00": "crack",
                "D10": "crack",
                "D20": "crack",
            },
            "default_class": None,
        }
        config_path = tmp_path / "many_to_one.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        mapper = TargetMapper(config_path)
        result = mapper.reverse_map("crack")
        assert sorted(result) == ["D00", "D10", "D20"]

    def test_reverse_no_match(self, sample_config):
        mapper = TargetMapper(sample_config)
        result = mapper.reverse_map("nonexistent")
        assert result == []


class TestValidate:
    def test_valid_taxonomy(self, sample_config):
        mapper = TargetMapper(sample_config)
        # Should not raise
        mapper.validate(["bache", "fisura_longitudinal", "fisura_transversal", "piel_de_cocodrilo"])

    def test_invalid_target_raises(self, sample_config):
        mapper = TargetMapper(sample_config)
        with pytest.raises(ConfigurationError) as exc_info:
            mapper.validate(["bache", "fisura_longitudinal"])
        # Should mention the invalid classes
        assert "fisura_transversal" in str(exc_info.value) or "piel_de_cocodrilo" in str(exc_info.value)

    def test_all_invalid_listed(self, tmp_path):
        config = {
            "taxonomy": [],
            "mappings": {
                "A": "x",
                "B": "y",
                "C": "x",
            },
            "default_class": None,
        }
        config_path = tmp_path / "invalid.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        mapper = TargetMapper(config_path)
        with pytest.raises(ConfigurationError) as exc_info:
            mapper.validate(["z"])
        # Both x and y should be reported
        assert len(exc_info.value.violations) == 2
