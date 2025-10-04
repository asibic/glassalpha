from __future__ import annotations

from glassalpha.core.registry import ModelRegistry

# Ensure entry points are visible to the registry
ModelRegistry.discover()

# Register built-ins that should always exist for tests and quickstart flows
from .passthrough import PassThroughModel  # noqa: E402

ModelRegistry.register("passthrough", PassThroughModel)

# Import tabular models to trigger their registration (they register themselves)
# These imports are conditional - they only succeed if dependencies are installed
try:
    from .tabular.lightgbm import LightGBMWrapper  # noqa: F401
except ImportError:
    pass

try:
    from .tabular.xgboost import XGBoostWrapper  # noqa: F401
except ImportError:
    pass

try:
    from .tabular.sklearn import LogisticRegressionWrapper, SklearnGenericWrapper  # noqa: F401
except ImportError:
    pass

__all__ = ["ModelRegistry", "PassThroughModel"]
