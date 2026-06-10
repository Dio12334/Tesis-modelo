"""Model registry and base detector interface for the Road Damage Evaluation Framework."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

from model.exceptions import ConfigurationError, ModelNotFoundError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch


class BaseDetector(ABC):
    """Base class for all detection models.

    All detection model implementations must inherit from this class
    and implement the abstract methods.
    """

    @abstractmethod
    def forward(self, images: "torch.Tensor") -> List[dict]:
        """Run forward pass on a batch of images.

        Args:
            images: Batch of images as a torch.Tensor.

        Returns:
            List of dicts per image, each containing:
                - boxes: Tensor of shape (N, 4) with bounding box coordinates
                - labels: Tensor of shape (N,) with class indices
                - scores: Tensor of shape (N,) with confidence scores
        """
        ...

    @abstractmethod
    def get_config_schema(self) -> dict:
        """Return required configuration parameters.

        Returns:
            Dict describing required config params. Keys are parameter names,
            values are dicts with 'type' and 'required' fields.
            Example: {"backbone_size": {"type": "str", "required": True}}
        """
        ...

    @abstractmethod
    def load_checkpoint(self, path: Path) -> None:
        """Load model weights from a checkpoint file.

        Args:
            path: Path to the checkpoint file.
        """
        ...

    @abstractmethod
    def save_checkpoint(self, path: Path) -> None:
        """Save model weights to a checkpoint file.

        Args:
            path: Path where the checkpoint will be saved.
        """
        ...

    def _underlying_model(self):
        """Return the wrapped model object, preferring ``_model`` then ``model``.

        Returns:
            The wrapped ``nn.Module``-like object held in the ``_model``
            attribute when present and non-None, otherwise the object held in
            the ``model`` attribute, or ``None`` when neither is present.
        """
        if getattr(self, "_model", None) is not None:
            return self._model
        return getattr(self, "model", None)

    def set_train_mode(self) -> None:
        """Set the underlying model to training mode.

        Default implementation that operates on whichever of ``_model`` or
        ``model`` is present. A no-op when neither attribute is set or the
        wrapped object does not expose a ``train`` method.
        """
        m = self._underlying_model()
        if m is not None and hasattr(m, "train"):
            m.train()

    def set_eval_mode(self) -> None:
        """Set the underlying model to evaluation mode.

        Default implementation that operates on whichever of ``_model`` or
        ``model`` is present. A no-op when neither attribute is set or the
        wrapped object does not expose an ``eval`` method.
        """
        m = self._underlying_model()
        if m is not None and hasattr(m, "eval"):
            m.eval()

    def to_device(self, device) -> None:
        """Move the wrapped model to ``device``.

        Default implementation that operates on whichever of ``_model`` or
        ``model`` is present. A no-op when neither attribute is set or the
        wrapped object does not expose a ``to`` method. Subclasses that manage
        their own device placement (Requirement 2.6) override this method and
        the override leaves the model on the requested device.

        Args:
            device: The target device to move the wrapped model to.
        """
        m = self._underlying_model()
        if m is not None and hasattr(m, "to"):
            m.to(device)
        if hasattr(self, "_device"):
            self._device = device


class ModelRegistry:
    """Singleton registry for detection model classes.

    Provides decorator-based registration and factory-style instantiation
    of detection models.
    """

    _models: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class under a given name.

        Args:
            name: Unique string identifier for the model.

        Returns:
            Decorator function that registers the model class.
        """

        def decorator(model_cls):
            cls._models[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, config: dict) -> BaseDetector:
        """Instantiate a registered model with the given configuration.

        Args:
            name: Registered model identifier.
            config: Configuration dict for the model.

        Returns:
            An instance of the registered model class.

        Raises:
            ModelNotFoundError: If the model name is not registered.
            ConfigurationError: If required configuration parameters are missing.
        """
        if name not in cls._models:
            raise ModelNotFoundError(name, cls.list_models())

        model_cls = cls._models[name]

        # Validate config against the model's schema
        # Instantiate temporarily to get schema, or call as classmethod/staticmethod
        # We need to check schema before instantiation. Use a temporary approach:
        # If the model class has get_config_schema as a classmethod or we can inspect it
        schema = cls._get_schema(model_cls)
        missing_params = cls._validate_config(config, schema)

        if missing_params:
            raise ConfigurationError(
                [f"Missing required parameter: {param}" for param in missing_params]
            )

        return model_cls(config)

    @classmethod
    def list_models(cls) -> List[str]:
        """Return all registered model identifiers in sorted order.

        Returns:
            Sorted list of registered model name strings.
        """
        return sorted(cls._models.keys())

    @classmethod
    def _get_schema(cls, model_cls: type) -> dict:
        """Get the config schema from a model class.

        Attempts to call get_config_schema as a class method or on a
        temporary basis. If the model defines it as a classmethod or
        staticmethod, call it directly. Otherwise, inspect the method.

        Args:
            model_cls: The model class to get schema from.

        Returns:
            Config schema dict.
        """
        # Check if get_config_schema can be called without an instance
        # (e.g., if it's overridden as a classmethod/staticmethod in a concrete class)
        # For ABC subclasses, we need a way to get schema without instantiation.
        # Convention: if the class defines get_config_schema, we try calling it
        # on the class. If that fails, we return an empty schema.
        try:
            # Try calling as unbound method with None (won't work for most cases)
            # Instead, check if it's defined and not abstract
            method = getattr(model_cls, "get_config_schema", None)
            if method is None:
                return {}

            # If it's a classmethod or staticmethod, call directly
            if isinstance(
                model_cls.__dict__.get("get_config_schema"), (classmethod, staticmethod)
            ):
                return model_cls.get_config_schema()

            # For regular methods, create a minimal instance without calling __init__
            instance = object.__new__(model_cls)
            try:
                return instance.get_config_schema()
            except AttributeError:
                # get_config_schema() reads __init__ state — it should be a @classmethod
                logger.warning(
                    "get_config_schema() on %s accesses instance attributes; "
                    "declare it as a @classmethod to avoid this.",
                    model_cls.__name__,
                )
                return {}
        except TypeError:
            return {}

    @classmethod
    def _validate_config(cls, config: dict, schema: dict) -> List[str]:
        """Validate config against schema, returning missing required params.

        Args:
            config: The configuration dict to validate.
            schema: The schema dict describing required parameters.

        Returns:
            List of missing required parameter names.
        """
        missing = []
        for param_name, param_spec in schema.items():
            if isinstance(param_spec, dict) and param_spec.get("required", False):
                if param_name not in config:
                    missing.append(param_name)
            elif param_spec is True:
                # Simple schema format: {"param_name": True} means required
                if param_name not in config:
                    missing.append(param_name)
        return missing

    @classmethod
    def reset(cls) -> None:
        """Clear all registered models. Useful for testing."""
        cls._models = {}
