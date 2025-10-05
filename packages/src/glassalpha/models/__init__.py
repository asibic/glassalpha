from __future__ import annotations

from glassalpha.core.registry import ModelRegistry

# Ensure entry points are visible to the registry
ModelRegistry.discover()

# Import PassThroughModel from core (it's already registered by noop_components)
from glassalpha.core.noop_components import PassThroughModel  # noqa: E402

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
