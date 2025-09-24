"""Centralized random seed management for reproducible audits.

This module provides comprehensive seed management across all randomness
sources in the audit pipeline: Python random, NumPy, scikit-learn, and
optional deep learning frameworks.
"""

import logging
import os
import random
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SeedManager:
    """Centralized manager for all random seeds in the audit pipeline."""

    def __init__(self, master_seed: int = 42):
        """Initialize seed manager with master seed.

        Args:
            master_seed: Primary seed for all randomness

        """
        self.master_seed = master_seed
        self.component_seeds: dict[str, int] = {}
        self._original_states: dict[str, Any] = {}

    def get_seed(self, component: str) -> int:
        """Get deterministic seed for a specific component.

        Args:
            component: Name of component needing seed (e.g., 'model', 'explainer')

        Returns:
            Deterministic seed derived from master seed

        """
        if component not in self.component_seeds:
            # Generate deterministic component seed from master seed and component name
            component_hash = hash(component + str(self.master_seed)) % (2**31)
            self.component_seeds[component] = abs(component_hash)
            logger.debug(f"Generated seed for '{component}': {self.component_seeds[component]}")

        return self.component_seeds[component]

    def set_all_seeds(self, seed: int | None = None) -> None:
        """Set all random seeds for reproducible execution.

        Args:
            seed: Override master seed (default: use current master_seed)

        """
        if seed is not None:
            self.master_seed = seed
            self.component_seeds.clear()  # Clear cached seeds

        logger.info(f"Setting all random seeds to master seed: {self.master_seed}")

        # Set Python random seed
        random.seed(self.master_seed)

        # Set NumPy seed
        np.random.seed(self.master_seed)

        # Set sklearn seed via environment variable
        os.environ["PYTHONHASHSEED"] = str(self.master_seed)

        # Set optional ML framework seeds
        self._set_optional_framework_seeds()

    def _set_optional_framework_seeds(self) -> None:
        """Set seeds for optional ML frameworks if available."""
        # PyTorch
        try:
            import torch

            torch.manual_seed(self.master_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.master_seed)
                torch.cuda.manual_seed_all(self.master_seed)
                # For deterministic CUDA operations
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            logger.debug("Set PyTorch seeds")
        except ImportError:
            pass

        # TensorFlow
        try:
            import tensorflow as tf

            tf.random.set_seed(self.master_seed)
            logger.debug("Set TensorFlow seed")
        except ImportError:
            pass

        # XGBoost and LightGBM use random_state parameters directly
        # These are handled in model wrappers using get_seed()

    def save_random_states(self) -> None:
        """Save current random states for restoration."""
        self._original_states = {
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),
        }

        # Save optional framework states
        try:
            import torch

            self._original_states["torch_random"] = torch.get_rng_state()
            if torch.cuda.is_available():
                self._original_states["torch_cuda_random"] = torch.cuda.get_rng_state()
        except ImportError:
            pass

        logger.debug("Saved random states")

    def restore_random_states(self) -> None:
        """Restore previously saved random states."""
        if not self._original_states:
            logger.warning("No saved random states to restore")
            return

        # Restore Python and NumPy states
        if "python_random" in self._original_states:
            random.setstate(self._original_states["python_random"])

        if "numpy_random" in self._original_states:
            np.random.set_state(self._original_states["numpy_random"])

        # Restore optional framework states
        try:
            import torch

            if "torch_random" in self._original_states:
                torch.set_rng_state(self._original_states["torch_random"])
            if "torch_cuda_random" in self._original_states:
                torch.cuda.set_rng_state(self._original_states["torch_cuda_random"])
        except ImportError:
            pass

        logger.debug("Restored random states")

    def get_seeds_manifest(self) -> dict[str, Any]:
        """Get manifest of all seeds used in this session.

        Returns:
            Dictionary with seed information for audit trail

        """
        return {
            "master_seed": self.master_seed,
            "component_seeds": self.component_seeds.copy(),
            "framework_availability": {
                "torch": self._check_framework_availability("torch"),
                "tensorflow": self._check_framework_availability("tensorflow"),
            },
        }

    def _check_framework_availability(self, framework: str) -> bool:
        """Check if optional framework is available."""
        try:
            __import__(framework)
            return True
        except ImportError:
            return False


# Global seed manager instance
_global_seed_manager = SeedManager()


def set_global_seed(seed: int) -> None:
    """Set global random seed for entire application.

    Args:
        seed: Master seed for all randomness

    """
    _global_seed_manager.set_all_seeds(seed)


def get_component_seed(component: str) -> int:
    """Get deterministic seed for specific component.

    Args:
        component: Component name (e.g., 'model', 'explainer', 'data_split')

    Returns:
        Deterministic seed for component

    """
    return _global_seed_manager.get_seed(component)


@contextmanager
def with_seed(seed: int, restore_after: bool = True) -> Generator[None, None, None]:
    """Context manager for temporary seed setting.

    Args:
        seed: Temporary seed to use
        restore_after: Whether to restore original state after context

    Yields:
        Context with temporary seed set

    Example:
        >>> with with_seed(123):
        ...     # All randomness uses seed 123
        ...     random_data = np.random.rand(10)
        >>> # Original random state restored

    """
    if restore_after:
        _global_seed_manager.save_random_states()

    # Set temporary seed
    _global_seed_manager.set_all_seeds(seed)

    try:
        yield
    finally:
        if restore_after:
            _global_seed_manager.restore_random_states()


@contextmanager
def with_component_seed(component: str) -> Generator[int, None, None]:
    """Context manager for component-specific seeding.

    Args:
        component: Component name for seed generation

    Yields:
        Component seed value

    Example:
        >>> with with_component_seed('model_training') as seed:
        ...     model = XGBoost(random_state=seed)
        ...     model.fit(X, y)

    """
    seed = get_component_seed(component)

    with with_seed(seed, restore_after=True):
        yield seed


def ensure_reproducibility(func):
    """Decorator to ensure function runs with consistent seeding.

    Args:
        func: Function to wrap with seed management

    Returns:
        Wrapped function with seed management

    Example:
        @ensure_reproducibility
        def train_model(data):
            # This function will have consistent randomness
            return model.fit(data)

    """

    def wrapper(*args, **kwargs):
        # Save current state
        _global_seed_manager.save_random_states()

        try:
            # Ensure seeds are set
            _global_seed_manager.set_all_seeds()
            return func(*args, **kwargs)
        finally:
            # Restore original state
            _global_seed_manager.restore_random_states()

    return wrapper


def get_seeds_manifest() -> dict[str, Any]:
    """Get complete seed manifest for audit trail.

    Returns:
        Dictionary with all seed information

    """
    return _global_seed_manager.get_seeds_manifest()


def validate_deterministic_environment() -> dict[str, bool]:
    """Validate that environment supports deterministic execution.

    Returns:
        Dictionary with validation results for each component

    """
    validation_results = {}

    # Test Python random
    random.seed(42)
    val1 = random.random()
    random.seed(42)
    val2 = random.random()
    validation_results["python_random"] = val1 == val2

    # Test NumPy random
    np.random.seed(42)
    arr1 = np.random.rand(5)
    np.random.seed(42)
    arr2 = np.random.rand(5)
    validation_results["numpy_random"] = np.allclose(arr1, arr2)

    # Test optional frameworks
    try:
        import torch

        torch.manual_seed(42)
        tensor1 = torch.rand(5)
        torch.manual_seed(42)
        tensor2 = torch.rand(5)
        validation_results["torch"] = torch.allclose(tensor1, tensor2)
    except ImportError:
        validation_results["torch"] = None

    try:
        import tensorflow as tf

        tf.random.set_seed(42)
        tensor1 = tf.random.uniform([5])
        tf.random.set_seed(42)
        tensor2 = tf.random.uniform([5])
        validation_results["tensorflow"] = tf.reduce_all(tf.equal(tensor1, tensor2)).numpy()
    except ImportError:
        validation_results["tensorflow"] = None

    logger.info(f"Deterministic environment validation: {validation_results}")
    return validation_results


# Convenience aliases
seed_manager = _global_seed_manager
