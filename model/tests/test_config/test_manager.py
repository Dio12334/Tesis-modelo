"""Unit tests for ConfigManager."""

import os
from pathlib import Path

import pytest
import yaml

from model.config.manager import ConfigManager
from model.exceptions import ValidationError


@pytest.fixture
def manager():
    return ConfigManager()


@pytest.fixture
def sample_config():
    return {
        "experiment": {
            "name": "test_run",
            "model": {"type": "yolov6", "backbone_size": "nano"},
        },
        "training": {
            "epochs": 50,
            "batch_size": 16,
            "learning_rate": 0.01,
        },
    }


class TestLoad:
    def test_load_valid_yaml(self, manager, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("name: test\nvalue: 42\n")
        result = manager.load(config_file)
        assert result == {"name": "test", "value": 42}

    def test_load_nested_yaml(self, manager, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("parent:\n  child: value\n  number: 10\n")
        result = manager.load(config_file)
        assert result == {"parent": {"child": "value", "number": 10}}

    def test_load_nonexistent_file_raises(self, manager):
        with pytest.raises(ValidationError) as exc_info:
            manager.load(Path("/nonexistent/path/config.yaml"))
        assert "not found" in exc_info.value.schema_violations[0]

    def test_load_invalid_yaml_raises(self, manager, tmp_path):
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("key: [unclosed bracket")
        with pytest.raises(ValidationError) as exc_info:
            manager.load(config_file)
        assert "Invalid YAML" in exc_info.value.schema_violations[0]

    def test_load_empty_file_returns_empty_dict(self, manager, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        result = manager.load(config_file)
        assert result == {}

    def test_load_non_dict_top_level_raises(self, manager, tmp_path):
        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item1\n- item2\n")
        with pytest.raises(ValidationError) as exc_info:
            manager.load(config_file)
        assert "mapping" in exc_info.value.schema_violations[0]


class TestMerge:
    def test_merge_disjoint_keys(self, manager):
        parent = {"a": 1, "b": 2}
        child = {"c": 3, "d": 4}
        result = manager.merge(parent, child)
        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_merge_child_overrides(self, manager):
        parent = {"a": 1, "b": 2}
        child = {"b": 99}
        result = manager.merge(parent, child)
        assert result == {"a": 1, "b": 99}

    def test_merge_deep_nested(self, manager):
        parent = {"level1": {"level2": {"a": 1, "b": 2}}}
        child = {"level1": {"level2": {"b": 99, "c": 3}}}
        result = manager.merge(parent, child)
        assert result == {"level1": {"level2": {"a": 1, "b": 99, "c": 3}}}

    def test_merge_child_replaces_non_dict_with_dict(self, manager):
        parent = {"key": "string_value"}
        child = {"key": {"nested": True}}
        result = manager.merge(parent, child)
        assert result == {"key": {"nested": True}}

    def test_merge_preserves_parent_only_keys(self, manager):
        parent = {"keep_me": "yes", "shared": "parent"}
        child = {"shared": "child"}
        result = manager.merge(parent, child)
        assert result["keep_me"] == "yes"
        assert result["shared"] == "child"

    def test_merge_does_not_mutate_parent(self, manager):
        parent = {"a": {"b": 1}}
        child = {"a": {"b": 2}}
        manager.merge(parent, child)
        assert parent == {"a": {"b": 1}}


class TestValidate:
    def test_validate_passes_valid_config(self, manager):
        config = {"name": "test", "epochs": 10}
        schema = {
            "required": ["name", "epochs"],
            "properties": {
                "name": {"type": "str"},
                "epochs": {"type": "int", "min": 1},
            },
        }
        # Should not raise
        manager.validate(config, schema)

    def test_validate_missing_required_field(self, manager):
        config = {"name": "test"}
        schema = {"required": ["name", "epochs"]}
        with pytest.raises(ValidationError) as exc_info:
            manager.validate(config, schema)
        assert any("epochs" in v for v in exc_info.value.schema_violations)

    def test_validate_wrong_type(self, manager):
        config = {"epochs": "not_a_number"}
        schema = {"properties": {"epochs": {"type": "int"}}}
        with pytest.raises(ValidationError) as exc_info:
            manager.validate(config, schema)
        assert any("type" in v for v in exc_info.value.schema_violations)

    def test_validate_below_min(self, manager):
        config = {"epochs": 0}
        schema = {"properties": {"epochs": {"type": "int", "min": 1}}}
        with pytest.raises(ValidationError) as exc_info:
            manager.validate(config, schema)
        assert any("minimum" in v for v in exc_info.value.schema_violations)

    def test_validate_above_max(self, manager):
        config = {"rate": 2.0}
        schema = {"properties": {"rate": {"type": "float", "max": 1.0}}}
        with pytest.raises(ValidationError) as exc_info:
            manager.validate(config, schema)
        assert any("maximum" in v for v in exc_info.value.schema_violations)

    def test_validate_enum_violation(self, manager):
        config = {"optimizer": "RMSProp"}
        schema = {"properties": {"optimizer": {"type": "str", "enum": ["SGD", "Adam", "AdamW"]}}}
        with pytest.raises(ValidationError) as exc_info:
            manager.validate(config, schema)
        assert any("allowed" in v for v in exc_info.value.schema_violations)

    def test_validate_collects_all_violations(self, manager):
        config = {"name": 123, "epochs": -1}
        schema = {
            "required": ["name", "epochs", "model"],
            "properties": {
                "name": {"type": "str"},
                "epochs": {"type": "int", "min": 1},
            },
        }
        with pytest.raises(ValidationError) as exc_info:
            manager.validate(config, schema)
        # Should have at least 3 violations: missing 'model', wrong type for 'name', below min for 'epochs'
        assert len(exc_info.value.schema_violations) >= 3

    def test_validate_nested_schema(self, manager):
        config = {"training": {"epochs": 10}}
        schema = {
            "properties": {
                "training": {
                    "type": "dict",
                    "required": ["epochs", "batch_size"],
                    "properties": {
                        "epochs": {"type": "int", "min": 1},
                    },
                }
            }
        }
        with pytest.raises(ValidationError) as exc_info:
            manager.validate(config, schema)
        assert any("batch_size" in v for v in exc_info.value.schema_violations)


class TestSave:
    def test_save_and_reload(self, manager, tmp_path, sample_config):
        output_path = tmp_path / "output.yaml"
        manager.save(sample_config, output_path)
        loaded = manager.load(output_path)
        assert loaded == sample_config

    def test_save_creates_parent_dirs(self, manager, tmp_path, sample_config):
        output_path = tmp_path / "nested" / "dir" / "config.yaml"
        manager.save(sample_config, output_path)
        assert output_path.exists()

    def test_save_produces_valid_yaml(self, manager, tmp_path):
        config = {"list_val": [1, 2, 3], "nested": {"key": "value"}}
        output_path = tmp_path / "config.yaml"
        manager.save(config, output_path)
        with open(output_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded == config


class TestResolveEnvVars:
    def test_resolve_single_var(self, manager, monkeypatch):
        monkeypatch.setenv("MY_VAR", "hello")
        config = {"greeting": "${MY_VAR}"}
        result = manager.resolve_env_vars(config)
        assert result == {"greeting": "hello"}

    def test_resolve_multiple_vars_in_string(self, manager, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "8080")
        config = {"url": "http://${HOST}:${PORT}/api"}
        result = manager.resolve_env_vars(config)
        assert result == {"url": "http://localhost:8080/api"}

    def test_resolve_nested_config(self, manager, monkeypatch):
        monkeypatch.setenv("DB_HOST", "db.example.com")
        config = {"database": {"host": "${DB_HOST}", "port": 5432}}
        result = manager.resolve_env_vars(config)
        assert result == {"database": {"host": "db.example.com", "port": 5432}}

    def test_resolve_in_list(self, manager, monkeypatch):
        monkeypatch.setenv("ITEM", "resolved")
        config = {"items": ["${ITEM}", "static"]}
        result = manager.resolve_env_vars(config)
        assert result == {"items": ["resolved", "static"]}

    def test_unset_var_left_unchanged(self, manager, monkeypatch):
        monkeypatch.delenv("UNSET_VAR", raising=False)
        config = {"key": "${UNSET_VAR}"}
        result = manager.resolve_env_vars(config)
        assert result == {"key": "${UNSET_VAR}"}

    def test_resolve_non_string_values_unchanged(self, manager):
        config = {"number": 42, "flag": True, "nothing": None}
        result = manager.resolve_env_vars(config)
        assert result == {"number": 42, "flag": True, "nothing": None}

    def test_resolve_does_not_mutate_original(self, manager, monkeypatch):
        monkeypatch.setenv("VAR", "new_value")
        config = {"key": "${VAR}"}
        manager.resolve_env_vars(config)
        assert config == {"key": "${VAR}"}
