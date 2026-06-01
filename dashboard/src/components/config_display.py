"""Config display component for the Streamlit Results Dashboard.

Renders experiment configuration in structured expandable sections
rather than raw JSON, covering training, model, dataset, and augmentation configs.
"""

import streamlit as st
from data_loader import ExperimentRun


def _render_key_value_pairs(data: dict, exclude_keys: set[str] | None = None) -> None:
    """Render a dictionary as formatted key-value pairs.

    Args:
        data: Dictionary of config values to display.
        exclude_keys: Optional set of keys to skip (e.g., nested dicts handled separately).
    """
    if exclude_keys is None:
        exclude_keys = set()

    for key, value in data.items():
        if key in exclude_keys:
            continue
        # Format the key as a readable label
        label = key.replace("_", " ").title()
        if isinstance(value, dict):
            # Render nested dicts as indented sub-sections
            st.markdown(f"**{label}:**")
            for sub_key, sub_value in value.items():
                sub_label = sub_key.replace("_", " ").title()
                st.text(f"  {sub_label}: {sub_value}")
        elif isinstance(value, list):
            st.text(f"{label}: {value}")
        else:
            st.text(f"{label}: {value}")


def _render_training_config(training: dict) -> None:
    """Render training configuration parameters.

    Displays learning_rate, batch_size, epochs, optimizer, scheduler,
    and other training hyperparameters.

    Args:
        training: The training section of the config dict.
    """
    # Core training parameters to highlight
    core_params = {
        "learning_rate": "Learning Rate",
        "batch_size": "Batch Size",
        "epochs": "Epochs",
        "optimizer": "Optimizer",
        "scheduler": "Scheduler",
    }

    for key, label in core_params.items():
        if key in training:
            st.text(f"{label}: {training[key]}")

    # Additional training parameters (exclude core ones and nested dicts)
    exclude = set(core_params.keys()) | {"augmentation"}
    additional = {k: v for k, v in training.items() if k not in exclude and not isinstance(v, dict)}
    if additional:
        st.markdown("**Additional Parameters:**")
        for key, value in additional.items():
            label = key.replace("_", " ").title()
            st.text(f"  {label}: {value}")


def _render_model_config(model: dict) -> None:
    """Render model configuration parameters.

    Displays model type, input_size, and num_classes.

    Args:
        model: The model section of the config dict.
    """
    if "type" in model:
        st.text(f"Model Type: {model['type']}")

    model_config = model.get("config", {})
    if "input_size" in model_config:
        st.text(f"Input Size: {model_config['input_size']}")
    if "num_classes" in model_config:
        st.text(f"Num Classes: {model_config['num_classes']}")

    # Any additional model config params
    additional = {k: v for k, v in model_config.items() if k not in ("input_size", "num_classes")}
    for key, value in additional.items():
        label = key.replace("_", " ").title()
        st.text(f"{label}: {value}")


def _render_dataset_config(dataset: dict) -> None:
    """Render dataset configuration parameters.

    Displays dataset type, path, and class mapping.

    Args:
        dataset: The dataset section of the config dict.
    """
    if "type" in dataset:
        st.text(f"Dataset Type: {dataset['type']}")
    if "path" in dataset:
        st.text(f"Path: {dataset['path']}")

    # Additional dataset params
    additional = {k: v for k, v in dataset.items() if k not in ("type", "path")}
    for key, value in additional.items():
        label = key.replace("_", " ").title()
        st.text(f"{label}: {value}")


def _render_augmentation_config(augmentation: dict) -> None:
    """Render augmentation configuration parameters.

    Displays all enabled transforms and their settings.

    Args:
        augmentation: The augmentation section of the config dict.
    """
    for key, value in augmentation.items():
        label = key.replace("_", " ").title()
        if isinstance(value, bool):
            status = "Enabled" if value else "Disabled"
            st.text(f"{label}: {status}")
        elif isinstance(value, list):
            st.text(f"{label}: {value}")
        else:
            st.text(f"{label}: {value}")


def render_config(run: ExperimentRun) -> None:
    """Render training config as structured expandable sections.

    Displays the experiment configuration organized into four categories:
    Training, Model, Dataset, and Augmentation. Each category is rendered
    inside a Streamlit expander for a clean, collapsible layout.

    Handles missing config sections gracefully by showing an informational
    message when a section is not available.

    Args:
        run: The ExperimentRun whose configuration to display.
    """
    st.subheader("Experiment Configuration")

    config = run.config
    if not config:
        st.info("No configuration data available for this run.")
        return

    # Training Configuration
    training = config.get("training", {})
    with st.expander("Training Configuration", expanded=False):
        if training:
            _render_training_config(training)
        else:
            st.info("No training configuration available.")

    # Model Configuration
    model = config.get("model", {})
    with st.expander("Model Configuration", expanded=False):
        if model:
            _render_model_config(model)
        else:
            st.info("No model configuration available.")

    # Dataset Configuration
    dataset = config.get("dataset", {})
    with st.expander("Dataset Configuration", expanded=False):
        if dataset:
            _render_dataset_config(dataset)
        else:
            st.info("No dataset configuration available.")

    # Augmentation Configuration
    augmentation = training.get("augmentation", {})
    with st.expander("Augmentation Configuration", expanded=False):
        if augmentation:
            _render_augmentation_config(augmentation)
        else:
            st.info("No augmentation configuration available.")
