"""Configuration schema definitions for the road damage evaluation framework.

Defines the EXPERIMENT_SCHEMA used by ConfigManager.validate() to check
experiment configuration files against expected structure, types, and constraints.
"""

EXPERIMENT_SCHEMA: dict = {
    "required": ["name", "model", "dataset", "training", "evaluation", "output"],
    "properties": {
        "name": {
            "type": "str",
        },
        "model": {
            "type": "dict",
            "required": ["type"],
            "properties": {
                "type": {
                    "type": "str",
                },
                "config": {
                    "type": "dict",
                },
            },
        },
        "dataset": {
            "type": "dict",
            "required": ["type", "path"],
            "properties": {
                "type": {
                    "type": "str",
                },
                "path": {
                    "type": "str",
                },
                "country_filter": {
                    "type": "list",
                },
                "class_mapping": {
                    "type": "str",
                },
            },
        },
        "training": {
            "type": "dict",
            "required": ["epochs", "batch_size", "learning_rate", "optimizer"],
            "properties": {
                "epochs": {
                    "type": "int",
                    "min": 1,
                },
                "batch_size": {
                    "type": "int",
                    "min": 1,
                },
                "learning_rate": {
                    "type": "float",
                    "min": 0,
                },
                "optimizer": {
                    "type": "str",
                    "enum": ["SGD", "Adam", "AdamW"],
                },
            },
        },
        "evaluation": {
            "type": "dict",
            "properties": {
                "iou_thresholds": {
                    "type": "list",
                },
                "confidence_threshold": {
                    "type": "float",
                },
                "target_classes": {
                    "type": "list",
                },
            },
        },
        "output": {
            "type": "dict",
            "properties": {
                "checkpoint_dir": {
                    "type": "str",
                },
                "results_dir": {
                    "type": "str",
                },
                "log_dir": {
                    "type": "str",
                },
            },
        },
    },
}
