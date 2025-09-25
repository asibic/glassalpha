"""Main audit pipeline orchestrator.

This module provides the central AuditPipeline class that coordinates
all components from data loading through model analysis to generate
comprehensive audit results with full reproducibility tracking.
"""

import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import AuditConfig
from ..core.registry import MetricRegistry, ModelRegistry
from ..data import TabularDataLoader, TabularDataSchema
from ..utils import ManifestGenerator, get_component_seed, set_global_seed

logger = logging.getLogger(__name__)


@dataclass
class AuditResults:
    """Container for comprehensive audit results."""

    # Core results
    model_performance: dict[str, Any] = field(default_factory=dict)
    fairness_analysis: dict[str, Any] = field(default_factory=dict)
    drift_analysis: dict[str, Any] = field(default_factory=dict)
    explanations: dict[str, Any] = field(default_factory=dict)

    # Data information
    data_summary: dict[str, Any] = field(default_factory=dict)
    schema_info: dict[str, Any] = field(default_factory=dict)

    # Model information
    model_info: dict[str, Any] = field(default_factory=dict)
    selected_components: dict[str, Any] = field(default_factory=dict)

    # Audit metadata
    execution_info: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)

    # Success indicators
    success: bool = False
    error_message: str | None = None


class AuditPipeline:
    """Main pipeline for conducting comprehensive ML model audits."""

    def __init__(self, config: AuditConfig):
        """Initialize audit pipeline with configuration.

        Args:
            config: Validated audit configuration

        """
        self.config = config
        self.results = AuditResults()

        # Ensure all components are imported and registered
        self._ensure_components_loaded()

        # Initialize manifest generator
        self.manifest_generator = ManifestGenerator()
        self.manifest_generator.add_config(config.model_dump())

        # Component instances (will be populated during execution)
        self.data_loader = TabularDataLoader()
        self.model = None
        self.explainer = None
        self.selected_metrics = {}

        logger.info(f"Initialized audit pipeline with profile: {config.audit_profile}")

    def run(self, progress_callback: callable = None) -> AuditResults:
        """Execute the complete audit pipeline.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Comprehensive audit results

        """
        try:
            logger.info("Starting audit pipeline execution")

            # Step 1: Setup reproducibility
            self._setup_reproducibility()
            self._update_progress(progress_callback, "Reproducibility setup complete", 10)

            # Step 2: Load and validate data
            data, schema = self._load_data()
            self._update_progress(progress_callback, "Data loaded and validated", 20)

            # Step 3: Load/initialize model
            self.model = self._load_model(data, schema)
            self._update_progress(progress_callback, "Model loaded/trained", 35)

            # Step 4: Select and initialize explainer
            self.explainer = self._select_explainer()
            self._update_progress(progress_callback, "Explainer selected", 45)

            # Step 5: Generate explanations
            explanations = self._generate_explanations(data, schema)
            self._update_progress(progress_callback, "Explanations generated", 60)

            # Step 6: Compute metrics
            self._compute_metrics(data, schema)
            self._update_progress(progress_callback, "Metrics computed", 80)

            # Step 7: Finalize results and manifest
            self._finalize_results(explanations)
            self._update_progress(progress_callback, "Audit complete", 100)

            self.results.success = True
            logger.info("Audit pipeline completed successfully")

        except Exception as e:
            error_msg = f"Audit pipeline failed: {e!s}"
            logger.error(error_msg)
            logger.debug(f"Full traceback: {traceback.format_exc()}")

            self.results.success = False
            self.results.error_message = error_msg

            # Mark manifest as failed
            self.manifest_generator.mark_completed("failed", error_msg)

        return self.results

    def _setup_reproducibility(self) -> None:
        """Set up reproducible execution environment."""
        logger.info("Setting up reproducible execution environment")

        # Set global seed from config
        master_seed = self.config.reproducibility.random_seed if self.config.reproducibility else 42
        set_global_seed(master_seed)

        # Add seed information to manifest
        self.manifest_generator.add_seeds()

        logger.debug(f"Global seed set to {master_seed}")

    def _load_data(self) -> tuple[pd.DataFrame, TabularDataSchema]:
        """Load and validate dataset.

        Returns:
            Tuple of (data, schema)

        """
        logger.info("Loading and validating dataset")

        # Get data path from config
        data_path = Path(self.config.data.path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load schema if specified
        schema = None
        if self.config.data.schema_path:
            # For now, create schema from config
            # TODO: Implement schema loading from file
            pass

        # Create schema from data config
        if not schema:
            # Extract schema information from config
            target_col = self.config.data.target_column or "target"
            feature_cols = self.config.data.feature_columns or []

            # If no feature columns specified, use all except target
            if not feature_cols:
                # We'll need to load data first to get column names
                temp_data = pd.read_csv(data_path)
                feature_cols = [col for col in temp_data.columns if col != target_col]

            schema = TabularDataSchema(
                target=target_col,
                features=feature_cols,
                sensitive_features=self.config.data.protected_attributes,
            )

        # Load data
        data = self.data_loader.load(data_path, schema)

        # Validate schema
        self.data_loader.validate_schema(data, schema)

        # Store data information in results
        self.results.data_summary = self.data_loader.get_data_summary(data)
        self.results.schema_info = schema.model_dump()

        # Add dataset to manifest
        self.manifest_generator.add_dataset(
            "primary_dataset",
            data=data,
            file_path=data_path,
            target_column=schema.target,
            sensitive_features=schema.sensitive_features,
        )

        logger.info(f"Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")

        return data, schema

    def _load_model(self, data: pd.DataFrame, schema: TabularDataSchema) -> Any:
        """Load or train model.

        Args:
            data: Dataset for training
            schema: Data schema

        Returns:
            Trained/loaded model instance

        """
        logger.info("Loading/training model")

        # Get model configuration
        model_type = self.config.model.type
        model_path = getattr(self.config.model, "path", None)

        # Get model class from registry
        model_class = ModelRegistry.get(model_type)
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")

        # Initialize model with component seed
        model_seed = get_component_seed("model")

        if model_path and Path(model_path).exists():
            # Load existing model
            model = model_class.from_file(Path(model_path))
            logger.info(f"Loaded model from {model_path}")
        else:
            # Train new model
            logger.info("Training new model (model path not found or not specified)")

            # Extract features and target
            X, y, _ = self.data_loader.extract_features_target(data, schema)

            # Preprocess features for training (handle categorical variables)
            X_processed = self._preprocess_for_training(X)

            # Handle different model types for training
            if model_type == "xgboost":
                # Train XGBoost directly and wrap it
                import xgboost as xgb

                dtrain = xgb.DMatrix(X_processed, label=y, feature_names=list(X_processed.columns))

                # XGBoost parameters with seed
                params = {
                    "objective": "binary:logistic",
                    "max_depth": 6,
                    "eta": 0.1,
                    "seed": model_seed,
                    "random_state": model_seed,
                }

                # Train model
                trained_model = xgb.train(params, dtrain, num_boost_round=100)

                # Wrap in our wrapper
                model = model_class(model=trained_model)

            elif model_type in ["lightgbm", "logistic_regression", "sklearn_generic"]:
                # These wrappers may have fit methods or need different handling
                model = model_class()

                # Try to use fit method if available
                if hasattr(model, "fit"):
                    model.fit(X, y, random_state=model_seed)
                else:
                    # For wrappers that don't support direct fitting
                    logger.warning(f"Model type {model_type} doesn't support direct training in pipeline")
                    model = model_class()
            else:
                # Default approach
                model = model_class()
                if hasattr(model, "fit"):
                    model.fit(X, y)

            logger.info("Model training completed")

        # Store model information
        feature_importance = {}
        if hasattr(model, "get_feature_importance"):
            try:
                importance = model.get_feature_importance()
                if hasattr(importance, "to_dict"):
                    feature_importance = importance.to_dict()
                elif isinstance(importance, dict):
                    feature_importance = importance
                else:
                    feature_importance = dict(importance) if importance is not None else {}
            except Exception as e:
                logger.warning(f"Could not extract feature importance: {e}")
                feature_importance = {}

        self.results.model_info = {
            "type": model_type,
            "capabilities": model.get_capabilities() if hasattr(model, "get_capabilities") else {},
            "feature_importance": feature_importance,
        }

        # Add model to manifest
        self.manifest_generator.add_component(
            "model",
            model_type,
            model,
            config=self.config.model.model_dump() if self.config.model else {},
        )

        return model

    def _select_explainer(self) -> Any:
        """Select best explainer based on model capabilities and configuration.

        Returns:
            Selected explainer instance

        """
        logger.info("Selecting explainer based on model capabilities")

        # Import the explainer registry to ensure it's available
        from ..explain.registry import ExplainerRegistry

        # First try automatic compatibility detection
        explainer_class = ExplainerRegistry.find_compatible(self.model)
        selected_name = None

        if explainer_class:
            # Found compatible explainer via automatic detection
            if explainer_class.__name__ == "TreeSHAPExplainer":
                selected_name = "treeshap"
            elif explainer_class.__name__ == "KernelSHAPExplainer":
                selected_name = "kernelshap"
            else:
                selected_name = "auto_detected"

            logger.info(f"Auto-detected compatible explainer: {selected_name}")
        else:
            # No automatic compatibility found, try explicit priority order
            logger.debug("No automatic compatibility found, trying priority order")

            # Get explainer priorities from config
            explainer_priorities = []
            if hasattr(self.config, "explainers") and hasattr(self.config.explainers, "priority"):
                explainer_priorities = self.config.explainers.priority
            else:
                # Use default priority order (but exclude noop - it should raise if nothing found)
                explainer_priorities = ["treeshap", "kernelshap"]

            # Try each explainer in priority order
            for explainer_name in explainer_priorities:
                try:
                    candidate_class = ExplainerRegistry.get(explainer_name)
                    if candidate_class:
                        # Check if this explainer is compatible with the model
                        instance = candidate_class()
                        if hasattr(instance, "is_compatible") and instance.is_compatible(self.model):
                            explainer_class = candidate_class
                            selected_name = explainer_name
                            logger.debug(f"Found compatible explainer: {explainer_name}")
                            break
                        logger.debug(f"Explainer {explainer_name} not compatible with model")
                except KeyError:
                    logger.debug(f"Explainer {explainer_name} not available in registry")
                    continue

        # If no explainer found, raise error (required by tests)
        if not explainer_class:
            raise RuntimeError("No compatible explainer found")

        # Create explainer instance
        selected_explainer = explainer_class()
        logger.info(f"Selected explainer: {selected_name}")

        # Store selection info
        self.results.selected_components["explainer"] = {
            "name": selected_name,
            "capabilities": getattr(selected_explainer, "capabilities", {}),
        }

        # Add to manifest
        self.manifest_generator.add_component(
            "explainer",
            selected_name,
            selected_explainer,
            priority=getattr(selected_explainer, "priority", None),
        )

        return selected_explainer

    def _generate_explanations(self, data: pd.DataFrame, schema: TabularDataSchema) -> dict[str, Any]:
        """Generate model explanations.

        Args:
            data: Dataset
            schema: Data schema

        Returns:
            Dictionary with explanation results

        """
        logger.info("Generating model explanations")

        # Extract features for explanation
        X, y, _ = self.data_loader.extract_features_target(data, schema)

        # Preprocess features same way as training
        X_processed = self._preprocess_for_training(X)

        # Generate explanations with explainer seed
        with self._get_seeded_context("explainer"):
            # Fit explainer with model and background data (use sample of data as background)
            background_sample = X_processed.sample(n=min(100, len(X_processed)), random_state=42)
            self.explainer.fit(self.model, background_sample, feature_names=list(X_processed.columns))

            # Generate explanations
            explanations = self.explainer.explain(X_processed)

        # Process explanations
        explanation_results = {
            "global_importance": explanations.get("global_importance", {}),
            "local_explanations_sample": explanations.get("local_explanations", [])[:5],  # First 5 samples
            "summary_statistics": self._compute_explanation_stats(explanations),
        }

        # Store in results
        self.results.explanations = explanation_results

        logger.info("Explanation generation completed")

        return explanation_results

    def _compute_metrics(self, data: pd.DataFrame, schema: TabularDataSchema) -> None:
        """Compute all configured metrics.

        Args:
            data: Dataset
            schema: Data schema

        """
        logger.info("Computing audit metrics")

        # Extract data components
        X, y_true, sensitive_features = self.data_loader.extract_features_target(data, schema)

        # Use processed features for predictions (same as training)
        X_processed = self._preprocess_for_training(X)

        # Generate predictions
        y_pred = self.model.predict(X_processed)
        y_proba = None
        if hasattr(self.model, "predict_proba"):
            try:
                y_proba = self.model.predict_proba(X_processed)
                if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                    y_proba = y_proba[:, 1]  # Binary classification positive class
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")

        # Compute performance metrics
        self._compute_performance_metrics(y_true, y_pred, y_proba)

        # Compute fairness metrics if sensitive features available
        if sensitive_features is not None:
            self._compute_fairness_metrics(y_true, y_pred, y_proba, sensitive_features)

        # Compute drift metrics (placeholder for now)
        self._compute_drift_metrics(X, y_true)

        logger.info("Metrics computation completed")

    def _compute_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> None:
        """Compute performance metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities

        """
        logger.debug("Computing performance metrics")

        # Get available performance metrics
        all_metrics = MetricRegistry.get_all()
        performance_metrics = {}

        for name, metric_class in all_metrics.items():
            # Check if it's a performance metric
            if (hasattr(metric_class, "metric_type") and metric_class.metric_type == "performance") or name in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auc_roc",
                "classification_report",
            ]:
                performance_metrics[name] = metric_class

        results = {}
        for metric_name, metric_class in performance_metrics.items():
            try:
                metric = metric_class()

                # Determine what arguments the metric needs
                if metric_name in ["auc_roc"] and y_proba is not None:
                    result = metric.compute(y_true, y_proba)
                else:
                    result = metric.compute(y_true, y_pred)

                results[metric_name] = result
                logger.debug(f"Computed {metric_name}: {result}")

            except Exception as e:
                logger.warning(f"Failed to compute metric {metric_name}: {e}")
                results[metric_name] = {"error": str(e)}

        self.results.model_performance = results

    def _compute_fairness_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        sensitive_features: pd.DataFrame,
    ) -> None:
        """Compute fairness metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            sensitive_features: Sensitive attributes

        """
        logger.debug("Computing fairness metrics")

        # Get available fairness metrics
        all_metrics = MetricRegistry.get_all()
        fairness_metrics = {}

        for name, metric_class in all_metrics.items():
            # Check if it's a fairness metric
            if (hasattr(metric_class, "metric_type") and metric_class.metric_type == "fairness") or name in [
                "demographic_parity",
                "equal_opportunity",
                "equalized_odds",
                "predictive_parity",
            ]:
                fairness_metrics[name] = metric_class

        results = {}
        for metric_name, metric_class in fairness_metrics.items():
            try:
                metric = metric_class()

                # Compute for each sensitive feature
                for col in sensitive_features.columns:
                    sensitive_attr = sensitive_features[col].values
                    result = metric.compute(y_true, y_pred, sensitive_features=sensitive_attr)

                    if col not in results:
                        results[col] = {}
                    results[col][metric_name] = result

                    logger.debug(f"Computed {metric_name} for {col}: {result}")

            except Exception as e:
                logger.warning(f"Failed to compute fairness metric {metric_name}: {e}")
                if col not in results:
                    results[col] = {}
                results[col][metric_name] = {"error": str(e)}

        self.results.fairness_analysis = results

    def _compute_drift_metrics(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Compute drift metrics (placeholder implementation).

        Args:
            X: Features
            y: Target values

        """
        logger.debug("Computing drift metrics (placeholder)")

        # For now, just record that we would compute drift metrics
        # This requires reference data which we don't have in this context
        self.results.drift_analysis = {
            "status": "not_computed",
            "reason": "no_reference_data",
            "available_methods": ["psi", "kl_divergence", "ks_test"],
        }

    def _finalize_results(self, explanations: dict[str, Any]) -> None:
        """Finalize audit results and manifest.

        Args:
            explanations: Generated explanations

        """
        logger.info("Finalizing audit results and manifest")

        # Store execution information
        self.results.execution_info = {
            "config_hash": self.manifest_generator.manifest.config_hash,
            "audit_profile": self.config.audit_profile,
            "strict_mode": self.config.strict_mode,
            "components_used": len(self.manifest_generator.manifest.selected_components),
        }

        # Add result hashes to manifest
        from ..utils import hash_object

        if self.results.model_performance:
            self.manifest_generator.add_result_hash("performance_metrics", hash_object(self.results.model_performance))

        if self.results.fairness_analysis:
            self.manifest_generator.add_result_hash("fairness_metrics", hash_object(self.results.fairness_analysis))

        if explanations:
            self.manifest_generator.add_result_hash("explanations", hash_object(explanations))

        # Finalize manifest
        final_manifest = self.manifest_generator.finalize()
        self.results.manifest = final_manifest.model_dump() if hasattr(final_manifest, "model_dump") else final_manifest

        logger.info("Audit results finalized")

    def _compute_explanation_stats(self, explanations: dict[str, Any]) -> dict[str, Any]:
        """Compute summary statistics for explanations.

        Args:
            explanations: Raw explanation results

        Returns:
            Summary statistics

        """
        stats = {}

        if "global_importance" in explanations:
            importance = explanations["global_importance"]
            if isinstance(importance, dict):
                values = list(importance.values())
                if values:
                    stats["mean_importance"] = float(np.mean(values))
                    stats["std_importance"] = float(np.std(values))
                    stats["top_features"] = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        return stats

    def _get_seeded_context(self, component_name: str):
        """Get seeded context manager for component.

        Args:
            component_name: Name of component for seed generation

        Returns:
            Context manager with component seed

        """
        from ..utils import with_component_seed

        return with_component_seed(component_name)

    def _update_progress(self, callback: callable, message: str, progress: int) -> None:
        """Update progress if callback provided.

        Args:
            callback: Optional progress callback
            message: Progress message
            progress: Progress percentage (0-100)

        """
        if callback:
            callback(progress, message)

        logger.debug(f"Progress: {progress}% - {message}")

    def _preprocess_for_training(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for model training.

        Args:
            X: Raw features DataFrame

        Returns:
            Processed features suitable for training

        """
        X_processed = X.copy()

        # Handle categorical columns by label encoding
        for col in X_processed.columns:
            if X_processed[col].dtype == "object":
                # Simple label encoding
                unique_values = X_processed[col].unique()
                value_map = {val: idx for idx, val in enumerate(unique_values)}
                X_processed[col] = X_processed[col].map(value_map)
                logger.debug(f"Label encoded column '{col}': {value_map}")

        # Ensure all columns are numeric
        for col in X_processed.columns:
            if X_processed[col].dtype == "object":
                X_processed[col] = pd.to_numeric(X_processed[col], errors="coerce")
                # Fill any NaN values with 0
                X_processed[col] = X_processed[col].fillna(0)

        return X_processed

    def _ensure_components_loaded(self) -> None:
        """Ensure all required components are imported and registered."""
        try:
            # Import model modules to trigger registration
            from ..explain.shap import kernel, tree  # noqa: F401
            from ..metrics.fairness import bias_detection  # noqa: F401
            from ..metrics.performance import classification  # noqa: F401
            from ..models.tabular import lightgbm, sklearn, xgboost  # noqa: F401

            logger.debug("All component modules imported and registered")
        except ImportError as e:
            logger.warning(f"Some components could not be imported: {e}")


def run_audit_pipeline(config: AuditConfig, progress_callback: callable = None) -> AuditResults:
    """Convenience function to run audit pipeline.

    Args:
        config: Validated audit configuration
        progress_callback: Optional progress callback function

    Returns:
        Audit results

    """
    pipeline = AuditPipeline(config)
    return pipeline.run(progress_callback)
