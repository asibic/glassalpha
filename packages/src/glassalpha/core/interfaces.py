"""Core protocol definitions for Glass Alpha components.

These protocols define the interfaces that all implementations must follow,
enabling the plugin architecture and future extensibility.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import numpy as np
import pandas as pd


@runtime_checkable
class ModelInterface(Protocol):
    """Protocol for all model wrappers."""
    
    # Required class attributes
    capabilities: Dict[str, Any]
    version: str
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for input data.
        
        Args:
            X: Input features as DataFrame
            
        Returns:
            Array of predictions
        """
        ...
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions for input data.
        
        Args:
            X: Input features as DataFrame
            
        Returns:
            Array of prediction probabilities
        """
        ...
    
    def get_model_type(self) -> str:
        """Return the model type identifier."""
        ...
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return model capabilities for plugin selection."""
        ...


@runtime_checkable
class ExplainerInterface(Protocol):
    """Protocol for all explainer implementations."""
    
    # Required class attributes
    capabilities: Dict[str, Any]
    version: str
    priority: int  # Higher = preferred
    
    def explain(
        self, 
        model: ModelInterface, 
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Generate explanations for the model.
        
        Args:
            model: Model to explain
            X: Input data
            y: Optional target values
            
        Returns:
            Dictionary containing explanation results
        """
        ...
    
    def supports_model(self, model: ModelInterface) -> bool:
        """Check if this explainer supports the given model.
        
        Args:
            model: Model to check compatibility
            
        Returns:
            True if model is supported
        """
        ...
    
    def get_explanation_type(self) -> str:
        """Return the type of explanation provided."""
        ...


@runtime_checkable
class MetricInterface(Protocol):
    """Protocol for all metric implementations."""
    
    # Required class attributes
    metric_type: str  # 'performance', 'fairness', 'drift', etc.
    version: str
    
    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Compute metric values.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            sensitive_features: Optional features for fairness metrics
            
        Returns:
            Dictionary of metric names to values
        """
        ...
    
    def get_metric_names(self) -> List[str]:
        """Return list of metric names computed."""
        ...
    
    def requires_sensitive_features(self) -> bool:
        """Check if metric requires sensitive features."""
        ...


@runtime_checkable
class DataInterface(Protocol):
    """Protocol for data handling across modalities."""
    
    # Required class attributes
    modality: str  # 'tabular', 'text', 'image', etc.
    version: str
    
    def load(self, path: str) -> Any:
        """Load data from path.
        
        Args:
            path: Path to data file
            
        Returns:
            Loaded data in appropriate format
        """
        ...
    
    def validate_schema(self, data: Any, schema: Dict[str, Any]) -> bool:
        """Validate data against schema.
        
        Args:
            data: Data to validate
            schema: Schema specification
            
        Returns:
            True if valid
        """
        ...
    
    def compute_hash(self, data: Any) -> str:
        """Compute deterministic hash of data.
        
        Args:
            data: Data to hash
            
        Returns:
            Hash string
        """
        ...


@runtime_checkable
class AuditProfileInterface(Protocol):
    """Protocol for audit profiles that define component sets."""
    
    # Required class attributes
    name: str
    version: str
    
    def get_compatible_models(self) -> List[str]:
        """Return list of compatible model types."""
        ...
    
    def get_required_metrics(self) -> List[str]:
        """Return list of required metric types."""
        ...
    
    def get_optional_metrics(self) -> List[str]:
        """Return list of optional metric types."""
        ...
    
    def get_report_template(self) -> str:
        """Return name of report template to use."""
        ...
    
    def get_explainer_priority(self) -> List[str]:
        """Return ordered list of preferred explainers."""
        ...
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration for this profile.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid for this profile
        """
        ...
