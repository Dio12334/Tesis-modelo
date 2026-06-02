"""Configuration schema definitions for the road damage evaluation framework.

Defines the EXPERIMENT_SCHEMA used by ConfigManager.validate() to check
experiment configuration files against expected structure, types, and constraints.

Also defines model-specific schemas for validating model.config sections.
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


# Model-specific config schemas keyed by model type
YOLO26_MODEL_CONFIG_SCHEMA: dict = {
    "required": [
        "model_size",
        "num_classes",
        "end2end",
        "confidence_threshold",
        "iou_threshold",
    ],
    "properties": {
        "model_size": {
            "type": "str",
            "enum": ["n", "s", "m", "l", "x"],
        },
        "num_classes": {
            "type": "int",
            "min": 1,
            "max": 1000,
        },
        "end2end": {
            "type": "bool",
        },
        "confidence_threshold": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
        },
        "iou_threshold": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
        },
        "pretrained_weights": {
            "type": "str",
        },
        "freeze_backbone": {
            "type": "bool",
        },
        "freeze_layers": {
            "type": "int",
            "min": 0,
        },
    },
}

MODEL_CONFIG_SCHEMAS: dict = {
    "yolo26": YOLO26_MODEL_CONFIG_SCHEMA,
}
