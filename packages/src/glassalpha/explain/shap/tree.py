"""TreeSHAP explainer for tree-based models.

TreeSHAP is an exact, efficient algorithm for computing SHAP values for tree-based
models. It provides local explanations and can aggregate to global feature importance.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import shap

from ...core.interfaces import ModelInterface
from ...core.registry import ExplainerRegistry

logger = logging.getLogger(__name__)


@ExplainerRegistry.register("treeshap", priority=100)
class TreeSHAPExplainer:
    """TreeSHAP explainer for tree-based models.
    
    This explainer uses the TreeSHAP algorithm to compute exact SHAP values
    for tree-based models like XGBoost, LightGBM, and Random Forest. It has
    the highest priority for these model types as it's both exact and efficient.
    """
    
    # Required class attributes for ExplainerInterface
    capabilities = {
        "supported_models": ["xgboost", "lightgbm", "random_forest", "decision_tree"],
        "explanation_type": "shap_values",
        "supports_local": True,
        "supports_global": True,
        "data_modality": "tabular"
    }
    version = "1.0.0"
    priority = 100  # Highest priority for tree models
    
    def __init__(self, check_additivity: bool = False):
        """Initialize TreeSHAP explainer.
        
        Args:
            check_additivity: Whether to check SHAP value additivity
        """
        self.check_additivity = check_additivity
        self.explainer = None
        self.base_value = None
        logger.info("TreeSHAPExplainer initialized")
    
    def explain(
        self,
        model: ModelInterface,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Generate SHAP explanations for the model.
        
        Args:
            model: Model to explain (must be tree-based)
            X: Input data to explain
            y: Optional target values (not used by TreeSHAP)
            
        Returns:
            Dictionary containing:
                - status: Success or error status
                - shap_values: SHAP values for each sample and feature
                - base_value: Expected value (baseline) for predictions
                - feature_importance: Global feature importance (mean absolute SHAP)
                - explainer_type: Type of explainer used
                - feature_names: Names of features
        """
        try:
            # Check if model is supported
            if not self.supports_model(model):
                return {
                    "status": "error",
                    "reason": f"Model type '{model.get_model_type()}' not supported by TreeSHAP",
                    "explainer_type": "treeshap"
                }
            
            # Get the underlying model object
            model_type = model.get_model_type()
            
            if model_type == "xgboost":
                # For XGBoost, use the Booster object directly
                if hasattr(model, 'model') and model.model is not None:
                    underlying_model = model.model
                else:
                    raise ValueError("XGBoost model not properly loaded")
            elif model_type == "lightgbm":
                # For LightGBM, use the Booster object
                if hasattr(model, 'model') and model.model is not None:
                    underlying_model = model.model
                else:
                    raise ValueError("LightGBM model not properly loaded")
            else:
                # For sklearn models, use the model directly
                underlying_model = model
            
            # Create TreeSHAP explainer
            logger.info(f"Creating TreeSHAP explainer for {model_type} model")
            self.explainer = shap.TreeExplainer(
                underlying_model,
                feature_perturbation="tree_path_dependent"  # More accurate for tree models
            )
            
            # Store base value
            if hasattr(self.explainer, 'expected_value'):
                self.base_value = self.explainer.expected_value
                # Handle multi-class case
                if isinstance(self.base_value, np.ndarray) and len(self.base_value) > 1:
                    # For binary classification, often we focus on the positive class
                    # For multi-class, we might need all values
                    logger.debug(f"Multi-class model with {len(self.base_value)} classes")
            else:
                self.base_value = 0.0
            
            # Calculate SHAP values
            logger.info(f"Computing SHAP values for {len(X)} samples")
            shap_values = self.explainer.shap_values(X)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                # Multi-class output - take the positive class for binary or all for multi
                if len(shap_values) == 2:
                    # Binary classification - use positive class
                    shap_values_array = shap_values[1]
                    base_value_scalar = self.base_value[1] if isinstance(self.base_value, np.ndarray) else self.base_value
                else:
                    # Multi-class - would need special handling
                    shap_values_array = shap_values
                    base_value_scalar = self.base_value
            else:
                # Single output (regression or binary with single output)
                shap_values_array = shap_values
                if isinstance(self.base_value, np.ndarray):
                    if len(self.base_value) == 2:
                        base_value_scalar = float(self.base_value[1])
                    else:
                        base_value_scalar = float(self.base_value[0])
                else:
                    base_value_scalar = float(self.base_value)
            
            # Calculate global feature importance (mean absolute SHAP values)
            if isinstance(shap_values_array, list):
                # Multi-class case - average across classes
                feature_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values_array], axis=0)
            else:
                feature_importance = np.abs(shap_values_array).mean(axis=0)
            
            # Create feature importance dictionary
            feature_names = list(X.columns)
            feature_importance_dict = {
                name: float(importance) 
                for name, importance in zip(feature_names, feature_importance)
            }
            
            # Sort by importance
            feature_importance_dict = dict(
                sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            
            logger.info("SHAP explanation completed successfully")
            
            return {
                "status": "success",
                "shap_values": shap_values_array,
                "base_value": base_value_scalar,
                "feature_importance": feature_importance,
                "feature_importance_dict": feature_importance_dict,
                "feature_names": feature_names,
                "explainer_type": "treeshap",
                "n_samples_explained": len(X),
                "n_features": len(feature_names)
            }
            
        except Exception as e:
            logger.error(f"Error in TreeSHAP explanation: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "reason": str(e),
                "explainer_type": "treeshap"
            }
    
    def supports_model(self, model: ModelInterface) -> bool:
        """Check if this explainer supports the given model.
        
        Args:
            model: Model to check compatibility
            
        Returns:
            True if model is a supported tree-based model
        """
        model_type = model.get_model_type()
        supported = model_type in self.capabilities["supported_models"]
        
        if supported:
            logger.debug(f"TreeSHAP supports model type: {model_type}")
        else:
            logger.debug(f"TreeSHAP does not support model type: {model_type}")
        
        return supported
    
    def get_explanation_type(self) -> str:
        """Return the type of explanation provided.
        
        Returns:
            String identifier for SHAP value explanations
        """
        return "shap_values"
    
    def __repr__(self) -> str:
        """String representation of the explainer."""
        return f"TreeSHAPExplainer(priority={self.priority}, version={self.version})"
