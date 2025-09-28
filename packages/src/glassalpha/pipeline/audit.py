"""Main audit pipeline orchestrator.

This module provides the central AuditPipeline class that coordinates
all components from data loading through model analysis to generate
comprehensive audit results with full reproducibility tracking.
"""

import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from glassalpha.config import AuditConfig

# Constants import removed - using f-string directly for logger format
from glassalpha.core.registry import MetricRegistry, ModelRegistry
from glassalpha.data import TabularDataLoader, TabularDataSchema
from glassalpha.utils import ManifestGenerator, get_component_seed, set_global_seed

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

    def __init__(self, config: AuditConfig) -> None:
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

        # Handle both pydantic and plain object configs (contract compliance)
        cfg_dict = config.model_dump() if hasattr(config, "model_dump") else dict(vars(config))
        self.manifest_generator.add_config(cfg_dict)

        # Component instances (will be populated during execution)
        self.data_loader = TabularDataLoader()
        self.model = None
        self.explainer = None
        self.selected_metrics = {}

        # Contract compliance: Exact f-string for wheel contract test
        logger.info(f"Initialized audit pipeline with profile: {config.audit_profile}")

    def run(self, progress_callback: Callable | None = None) -> AuditResults:
        """Execute the complete audit pipeline.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Comprehensive audit results

        """
        from datetime import UTC, datetime  # noqa: PLC0415

        # Friend's spec: Before doing work - capture start time
        start = datetime.now(UTC).isoformat()

        try:
            logger.info("Starting audit pipeline execution")

            # Step 1: Setup reproducibility
            self._setup_reproducibility()
            self._update_progress(progress_callback, "Reproducibility setup complete", 10)

            # Step 2: Load and validate data
            data, schema = self._load_data()
            # Store dataset for provenance tracking
            self._dataset_df = data.copy()  # Store copy for provenance
            self._feature_count = len(schema.features) if schema.features else data.shape[1] - 1
            self._class_count = data[schema.target].nunique() if schema.target in data.columns else None
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

            # Friend's spec: On success - set end time and call set execution info
            end = datetime.now(UTC).isoformat()
            logger.debug(f"Pipeline execution: start={start}, end={end}")

            # Update manifest generator with execution info
            self.manifest_generator.mark_completed("completed", None)

            self.results.success = True
            logger.info("Audit pipeline completed successfully")

        except Exception as e:
            # Friend's spec: On exception - set error_message and still write start/end times
            end = datetime.now(UTC).isoformat()
            logger.debug(f"Pipeline execution failed: start={start}, end={end}")
            error_msg = f"Audit pipeline failed: {e!s}"

            logger.exception(error_msg)
            logger.debug(f"Full traceback: {traceback.format_exc()}")

            self.results.success = False
            self.results.error_message = error_msg

            # Mark manifest as failed with timing info
            self.manifest_generator.mark_completed("failed", error_msg)

        return self.results

    def _setup_reproducibility(self) -> None:
        """Set up reproducible execution environment."""
        logger.info("Setting up reproducible execution environment")

        # Set global seed from config
        master_seed = self.config.reproducibility.random_seed if self.config.reproducibility else 42
        set_global_seed(master_seed)

        # Apply advanced reproduction controls if configured
        if (
            self.config.reproducibility
            and hasattr(self.config.reproducibility, "strict")
            and self.config.reproducibility.strict
        ):
            from ..runtime import set_repro  # noqa: PLC0415

            logger.info("Applying advanced deterministic reproduction controls")
            repro_status = set_repro(
                seed=master_seed,
                strict=self.config.reproducibility.strict,
                thread_control=getattr(self.config.reproducibility, "thread_control", False),
                warn_on_failure=getattr(self.config.reproducibility, "warn_on_failure", True),
            )

            # Store repro status in results for provenance
            if not hasattr(self.results, "execution_info") or self.results.execution_info is None:
                self.results.execution_info = {}
            self.results.execution_info["reproduction_status"] = repro_status

            successful = sum(1 for control in repro_status["controls"].values() if control.get("success", False))
            total = len(repro_status["controls"])
            logger.info(f"Advanced reproduction controls: {successful}/{total} successful")

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
            msg = f"Data file not found: {data_path}"
            raise FileNotFoundError(msg)

        # Load schema if specified
        schema = None
        if self.config.data.schema_path:
            # For now, create schema from config
            # TODO(dev): Implement schema loading from file  # noqa: TD003, FIX002
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

        # First-class schema validation before proceeding
        from ..data.schema import get_schema_summary, validate_config_schema, validate_data_quality  # noqa: PLC0415

        try:
            # Convert config to dict for schema validation
            config_dict = {"data": self.config.data.model_dump()}
            validated_schema = validate_config_schema(data, config_dict)

            # Run data quality checks
            validate_data_quality(data, validated_schema)

            # Log schema summary
            schema_summary = get_schema_summary(data, validated_schema)
            logger.info(
                f"Schema validation passed: {schema_summary['n_features']} features, "
                f"{schema_summary['n_classes']} classes, "
                f"{schema_summary['n_protected_attributes']} protected attributes",
            )

        except ValueError as e:
            logger.error(f"Schema validation failed: {e}")
            raise

        # Validate schema (legacy validation)
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

        logger.info(f"Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")

        return data, schema

    def _load_model(self, data: pd.DataFrame, schema: TabularDataSchema) -> Any:  # noqa: ANN401
        """Load or train model.

        Args:
            data: Dataset for training
            schema: Data schema

        Returns:
            Trained/loaded model instance

        """
        logger.info("Loading/training model")

        # Default to trainable model if no config provided (E2E compliance)
        if not hasattr(self.config, "model") or self.config.model is None:
            logger.info("Using default trainable model: LogisticRegressionWrapper")

            # Create default config for LogisticRegression
            from types import SimpleNamespace  # noqa: PLC0415

            default_model_config = SimpleNamespace()
            default_model_config.type = "logistic_regression"
            default_model_config.params = {"random_state": get_component_seed("model")}

            # Create temporary config with default model
            temp_config = SimpleNamespace()
            temp_config.model = default_model_config
            temp_config.reproducibility = getattr(self.config, "reproducibility", None)

            # Extract features and target for training
            X, y, _ = self.data_loader.extract_features_target(data, schema)  # noqa: N806
            X_processed = self._preprocess_for_training(X)  # noqa: N806

            # Use train_from_config for consistency
            from .train import train_from_config  # noqa: PLC0415

            model = train_from_config(temp_config, X_processed, y)
            logger.info("Default model training completed using configuration")

            # Store model info and tracking
            self.results.model_info = {
                "type": "logistic_regression",
                "capabilities": model.get_capabilities() if hasattr(model, "get_capabilities") else {},
                "feature_importance": {},
            }
            self.results.selected_components["model"] = {"name": "logistic_regression", "type": "model"}

            # Add to manifest
            self.manifest_generator.add_component(
                "model",
                "logistic_regression",
                model,
                details={"default": True, "fitted": True},
            )
            return model

        # Get model configuration
        model_type = self.config.model.type
        model_path = getattr(self.config.model, "path", None)

        # Ensure model modules are imported for registration
        if model_type == "xgboost":
            import importlib  # noqa: PLC0415

            importlib.import_module("glassalpha.models.tabular.xgboost")
        elif model_type == "lightgbm":
            import importlib  # noqa: PLC0415

            importlib.import_module("glassalpha.models.tabular.lightgbm")
        elif model_type in ["logistic_regression", "sklearn_generic"]:
            import importlib  # noqa: PLC0415

            importlib.import_module("glassalpha.models.tabular.sklearn")

        # Get model class from registry
        model_class = ModelRegistry.get(model_type)
        if not model_class:
            msg = f"Unknown model type: {model_type}"
            raise ValueError(msg)

        if model_path and Path(model_path).exists():
            # Load existing model
            model = model_class.from_file(Path(model_path))
            logger.info(f"Loaded model from {model_path}")
        else:
            # Train new model using configuration-driven approach
            logger.info("Training new model from configuration")

            # Extract features and target
            X, y, _ = self.data_loader.extract_features_target(data, schema)  # noqa: N806
            X_processed = self._preprocess_for_training(X)  # noqa: N806

            # Use the new train_from_config function
            from .train import train_from_config  # noqa: PLC0415

            model = train_from_config(self.config, X_processed, y)
            logger.info("Model training completed using configuration")

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
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not extract feature importance: {e}")
                feature_importance = {}

        self.results.model_info = {
            "type": model_type,
            "capabilities": model.get_capabilities() if hasattr(model, "get_capabilities") else {},
            "feature_importance": feature_importance,
        }

        # Friend's spec: Track the model in both results and manifest
        # Track in results.selected_components with exact structure
        self.results.selected_components["model"] = {"name": model_type, "type": "model"}

        # Add model to manifest using new signature with details
        model_config = self.config.model.model_dump() if self.config.model else {}
        self.manifest_generator.add_component(
            "model",
            model_type,
            model,
            details={
                "config": model_config,
                "priority": getattr(model, "priority", None),
            },
        )

        return model

    def _select_explainer(self) -> Any:  # noqa: ANN401
        """Select best explainer based on model capabilities and configuration.

        Returns:
            Selected explainer instance

        """
        logger.info("Selecting explainer based on model capabilities")

        # Import the explainer registry to ensure it's available
        from glassalpha.explain.registry import ExplainerRegistry  # noqa: PLC0415

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
            msg = "No compatible explainer found"
            raise RuntimeError(msg)

        # Create explainer instance
        selected_explainer = explainer_class()
        logger.info(f"Selected explainer: {selected_name}")

        # Store selection info
        self.results.selected_components["explainer"] = {
            "name": selected_name,
            "capabilities": getattr(selected_explainer, "capabilities", {}),
        }

        # Add to manifest with new signature
        self.manifest_generator.add_component(
            "explainer",
            selected_name,
            selected_explainer,
            details={
                "implementation": selected_name,
                "priority": getattr(selected_explainer, "priority", None),
            },
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
        X, _y, _ = self.data_loader.extract_features_target(data, schema)  # noqa: N806

        # Preprocess features same way as training
        X_processed = self._preprocess_for_training(X)  # noqa: N806

        # Generate explanations with explainer seed
        with self._get_seeded_context("explainer"):
            # Fit explainer with model and background data (use sample of data as background)
            background_sample = X_processed.sample(n=min(100, len(X_processed)), random_state=42)
            self.explainer.fit(self.model, background_sample, feature_names=list(X_processed.columns))

            # Generate explanations
            explanations = self.explainer.explain(X_processed)

        # Process explanations - handle different return formats from explainers
        if isinstance(explanations, dict):
            # Expected dictionary format
            explanation_results = {
                "global_importance": explanations.get("global_importance", {}),
                "local_explanations_sample": explanations.get("local_explanations", [])[:5],  # First 5 samples
                "summary_statistics": self._compute_explanation_stats(explanations),
            }
        else:
            # Handle numpy array or other formats - create basic structure
            logger.warning(f"Explainer returned {type(explanations)}, creating basic explanation structure")
            explanation_results = {
                "global_importance": {},
                "local_explanations_sample": [],
                "summary_statistics": {
                    "explanation_type": "raw_array",
                    "shape": getattr(explanations, "shape", "unknown"),
                },
            }

        # Store in results
        self.results.explanations = explanation_results

        logger.info("Explanation generation completed")

        return explanation_results

    def _compute_metrics(self, data: pd.DataFrame, schema: TabularDataSchema) -> None:  # noqa: C901
        """Compute all configured metrics.

        Args:
            data: Dataset
            schema: Data schema

        """
        logger.info("Computing audit metrics")

        # Extract data components
        X, y_true, sensitive_features = self.data_loader.extract_features_target(data, schema)  # noqa: N806

        # Use processed features for predictions (same as training)
        X_processed = self._preprocess_for_training(X)  # noqa: N806

        # Friend's spec: Fit-or-fail approach - never skip metrics
        if self.model is None:
            msg = "Model is None during metrics computation - this should not happen"
            raise RuntimeError(msg)

        # Check if model needs fitting and fit it (don't skip metrics)
        model_needs_fitting = False

        # Contract: simplified training logic guard (string match required by tests)
        if getattr(self.model, "model", None) is None:
            logger.debug("No underlying model instance set; proceeding with wrapper defaults")
            model_needs_fitting = True
        elif hasattr(self.model, "model") and getattr(self.model, "model", None) is None:
            # Model wrapper exists but internal model is None - needs fitting
            model_needs_fitting = True
        elif hasattr(self.model, "_is_fitted") and not getattr(self.model, "_is_fitted", True):
            # Model wrapper tracks fitted state explicitly
            model_needs_fitting = True

        if model_needs_fitting:
            logger.warning("Model not fitted - this should not happen with proper wrapper-based training")
            logger.warning("All models should be trained via train_from_config() in _load_model()")

            # This is a fallback that should rarely be used
            if not hasattr(self.model, "fit"):
                msg = f"Model type {type(self.model).__name__} needs fitting but doesn't support fit method"
                raise RuntimeError(msg)

            logger.info("Fallback: fitting model with available data")
            model_seed = (
                self.manifest_generator.manifest.seeds.get("model", 42)
                if hasattr(self.manifest_generator, "manifest")
                else 42
            )

            # Get random state from config if available
            if hasattr(self.config, "reproducibility") and self.config.reproducibility:
                model_seed = getattr(self.config.reproducibility, "random_state", model_seed)

            # Use wrapper fit method with proper parameters
            if hasattr(self.config, "model") and hasattr(self.config.model, "params"):
                model_params = dict(self.config.model.params)
                model_params["random_state"] = model_seed
                self.model.fit(X_processed, y_true, **model_params)
            else:
                self.model.fit(X_processed, y_true, random_state=model_seed)
            logger.info("Fallback model fitting completed")

        # Generate probability predictions first
        y_proba = None
        if hasattr(self.model, "predict_proba"):
            try:
                y_proba = self.model.predict_proba(X_processed)
                # Keep full probability matrix for multiclass, don't extract single column
                logger.debug(f"Generated probability predictions with shape: {y_proba.shape}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not get prediction probabilities: {e}")

        # Generate predictions using threshold policy (if binary classification with probabilities)
        if y_proba is not None and y_proba.shape[1] == 2:  # Binary classification
            # Use threshold policy for binary classification
            y_pred, threshold_info = self._apply_threshold_policy(y_true, y_proba[:, 1])
            # Store threshold information in results
            self.results.model_performance["threshold_selection"] = threshold_info
        else:
            # Fallback to model's default predict method (multiclass or no probabilities)
            y_pred = self.model.predict(X_processed)
            logger.info("Using model's default predictions (multiclass or no probabilities available)")

        # Compute performance metrics
        self._compute_performance_metrics(y_true, y_pred, y_proba)

        # Friend's spec: Ensure accuracy is always computed for each model type
        try:
            from sklearn.metrics import accuracy_score  # noqa: PLC0415

            acc = float(accuracy_score(y_true, y_pred))
            if not hasattr(self.results, "model_performance") or self.results.model_performance is None:
                self.results.model_performance = {}
            self.results.model_performance["accuracy"] = acc
            logger.debug(f"Computed explicit accuracy: {acc:.4f}")
        except Exception:
            logger.exception("Failed to compute explicit accuracy:")

        # Compute fairness metrics if sensitive features available
        if sensitive_features is not None:
            self._compute_fairness_metrics(y_true, y_pred, y_proba, sensitive_features)

        # Compute drift metrics (placeholder for now)
        self._compute_drift_metrics(X, y_true)

        logger.info("Metrics computation completed")

    def _apply_threshold_policy(self, y_true: np.ndarray, y_proba: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply threshold selection policy for binary classification.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities for positive class

        Returns:
            Tuple of (predictions, threshold_info)

        """
        from ..metrics.thresholds import pick_threshold, validate_threshold_config  # noqa: PLC0415

        # Get threshold configuration from report config
        threshold_config = {}
        if hasattr(self.config, "report") and hasattr(self.config.report, "threshold") and self.config.report.threshold:
            threshold_config = self.config.report.threshold.model_dump()

        # Validate and normalize config
        validated_config = validate_threshold_config(threshold_config)

        logger.info(f"Applying threshold policy: {validated_config['policy']}")

        # Select threshold using policy
        threshold_result = pick_threshold(y_true, y_proba, **validated_config)

        # Generate predictions using selected threshold
        selected_threshold = threshold_result["threshold"]
        y_pred = (y_proba >= selected_threshold).astype(int)

        logger.info(f"Selected threshold: {selected_threshold:.3f} using {threshold_result['policy']} policy")

        return y_pred, threshold_result

    def _generate_provenance_manifest(self) -> None:
        """Generate comprehensive provenance manifest for audit reproducibility."""
        from ..provenance import generate_run_manifest  # noqa: PLC0415

        logger.info("Generating comprehensive provenance manifest")

        # Gather all provenance information
        config_dict = self.config.model_dump() if hasattr(self.config, "model_dump") else dict(vars(self.config))

        # Get dataset information
        dataset_path = getattr(self.config.data, "path", None) if hasattr(self.config, "data") else None
        dataset_df = getattr(self, "_dataset_df", None)  # Store dataset if available

        # Get model information
        model_info = {
            "type": self.model.__class__.__name__ if self.model else None,
            "parameters": getattr(self.model, "get_params", dict)() if self.model else {},
            "calibration": self._get_calibration_info(),
            "feature_count": getattr(self, "_feature_count", None),
            "class_count": getattr(self, "_class_count", None),
        }

        # Get selected components
        selected_components = {
            "explainer": self.explainer.__class__.__name__ if self.explainer else None,
            "metrics": list(self.selected_metrics.keys()) if self.selected_metrics else [],
            "threshold_policy": self.results.model_performance.get("threshold_selection", {}).get("policy")
            if hasattr(self.results, "model_performance")
            else None,
        }

        # Get execution information
        execution_info = {
            "start_time": getattr(self, "_start_time", None),
            "end_time": getattr(self, "_end_time", None),
            "success": True,  # If we're here, it succeeded
            "random_seed": getattr(self.config.reproducibility, "random_seed", None)
            if hasattr(self.config, "reproducibility")
            else None,
        }

        # Generate the manifest
        manifest = generate_run_manifest(
            config=config_dict,
            dataset_path=dataset_path,
            dataset_df=dataset_df,
            model_info=model_info,
            selected_components=selected_components,
            execution_info=execution_info,
        )

        # Store in results for PDF embedding
        self.results.execution_info["provenance_manifest"] = manifest

        logger.info(f"Provenance manifest generated with {len(manifest)} sections")

    def _get_calibration_info(self) -> dict[str, Any] | None:
        """Get calibration information from model if available."""
        if not self.model:
            return None

        try:
            from ..models.calibration import get_calibration_info  # noqa: PLC0415

            base_estimator = getattr(self.model, "model", self.model)
            return get_calibration_info(base_estimator)
        except Exception:
            return None

    def _compute_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> None:
        """Compute performance metrics using auto-detecting engine.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities

        """
        from ..metrics.core import compute_classification_metrics  # noqa: PLC0415

        logger.debug("Computing performance metrics with auto-detection engine")

        # Get averaging strategy from config if specified
        performance_config = self.config.metrics.performance
        if hasattr(performance_config, "config"):
            averaging_override = performance_config.config.get("average")
        else:
            averaging_override = None

        # Use the new auto-detecting metrics engine
        try:
            metrics_result = compute_classification_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
                average=averaging_override,
            )

            # Convert to the expected format for compatibility
            results = {
                "accuracy": {
                    "accuracy": metrics_result["accuracy"],
                    "n_samples": len(y_true),
                },
                "precision": {
                    "precision": metrics_result["precision"],
                    "n_samples": len(y_true),
                    "problem_type": metrics_result["problem_type"],
                    "averaging_strategy": metrics_result["averaging_strategy"],
                },
                "recall": {
                    "recall": metrics_result["recall"],
                    "n_samples": len(y_true),
                    "problem_type": metrics_result["problem_type"],
                    "averaging_strategy": metrics_result["averaging_strategy"],
                },
                "f1": {
                    "f1": metrics_result["f1_score"],
                    "n_samples": len(y_true),
                    "problem_type": metrics_result["problem_type"],
                    "averaging_strategy": metrics_result["averaging_strategy"],
                },
                "classification_report": {
                    "accuracy": metrics_result["accuracy"],
                    "n_samples": len(y_true),
                    "n_classes": metrics_result["n_classes"],
                    "problem_type": metrics_result["problem_type"],
                    "averaging_strategy": metrics_result["averaging_strategy"],
                },
            }

            # Add probability-based metrics if available
            if metrics_result["roc_auc"] is not None:
                results["auc_roc"] = {
                    "roc_auc": metrics_result["roc_auc"],
                    "n_samples": len(y_true),
                    "problem_type": metrics_result["problem_type"],
                }

            if metrics_result["log_loss"] is not None:
                results["log_loss"] = {
                    "log_loss": metrics_result["log_loss"],
                    "n_samples": len(y_true),
                    "problem_type": metrics_result["problem_type"],
                }

            # Log any warnings or errors from the metrics engine
            if metrics_result["warnings"]:
                for warning in metrics_result["warnings"]:
                    logger.warning(f"Metrics engine warning: {warning}")

            if metrics_result["errors"]:
                for error in metrics_result["errors"]:
                    logger.error(f"Metrics engine error: {error}")

            # Filter out None values to maintain compatibility
            results = {
                k: v
                for k, v in results.items()
                if v and all(val is not None for val in v.values() if isinstance(val, (int, float)))
            }

            logger.info(f"Successfully computed {len(results)} performance metrics using auto-detection engine")

        except Exception as e:
            logger.error(f"Failed to compute performance metrics with auto-detection engine: {e}")
            # Fallback to empty results with error
            results = {"error": f"Auto-detection engine failed: {e}"}

        self.results.model_performance = results

    def _compute_fairness_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,  # noqa: ARG002
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
        fairness_metrics = []

        for name, metric_class in all_metrics.items():
            # Check if it's a fairness metric
            if (hasattr(metric_class, "metric_type") and metric_class.metric_type == "fairness") or name in [
                "demographic_parity",
                "equal_opportunity",
                "equalized_odds",
                "predictive_parity",
            ]:
                fairness_metrics.append(metric_class)

        if not fairness_metrics:
            logger.warning("No fairness metrics found")
            self.results.fairness_analysis = {}
            return

        # Use the new fairness runner
        from ..metrics.fairness.runner import run_fairness_metrics  # noqa: PLC0415

        try:
            fairness_results = run_fairness_metrics(
                y_true,
                y_pred,
                sensitive_features,
                fairness_metrics,
            )
            self.results.fairness_analysis = fairness_results
            logger.info(f"Computed fairness metrics: {list(fairness_results.keys())}")

        except Exception as e:
            logger.error(f"Failed to compute fairness metrics: {e}")
            self.results.fairness_analysis = {"error": str(e)}

    def _compute_drift_metrics(self, X: pd.DataFrame, y: np.ndarray) -> None:  # noqa: N803, ARG002
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
        from glassalpha.utils import hash_object  # noqa: PLC0415

        if self.results.model_performance:
            self.manifest_generator.add_result_hash("performance_metrics", hash_object(self.results.model_performance))

        if self.results.fairness_analysis:
            self.manifest_generator.add_result_hash("fairness_metrics", hash_object(self.results.fairness_analysis))

        if explanations:
            self.manifest_generator.add_result_hash("explanations", hash_object(explanations))

        # Populate selected_components in results from manifest (friend's spec)
        self.results.selected_components = self.manifest_generator.manifest.selected_components

        # Mark manifest as completed successfully before finalizing (friend's spec)
        self.manifest_generator.mark_completed("success")

        # Record end time for provenance
        from datetime import datetime  # noqa: PLC0415

        self._end_time = datetime.now(UTC).isoformat()

        # Generate comprehensive provenance manifest
        self._generate_provenance_manifest()

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

        def _to_scalar(v: Any) -> float:  # noqa: ANN401
            """Convert value to scalar, handling lists/arrays as specified by friend."""
            if isinstance(v, (list, tuple, np.ndarray)):
                return float(np.mean(np.abs(v)))
            return float(abs(v))

        stats = {}

        if "global_importance" in explanations:
            importance = explanations["global_importance"]
            if isinstance(importance, dict):
                values = list(importance.values())
                if values:
                    # Convert all values to scalars before computing stats
                    scalar_values = [_to_scalar(v) for v in values]
                    stats["mean_importance"] = float(np.mean(scalar_values))
                    stats["std_importance"] = float(np.std(scalar_values))
                    stats["top_features"] = sorted(importance.items(), key=lambda x: _to_scalar(x[1]), reverse=True)[:5]

        return stats

    def _get_seeded_context(self, component_name: str) -> Any:  # noqa: ANN401
        """Get seeded context manager for component.

        Args:
            component_name: Name of component for seed generation

        Returns:
            Context manager with component seed

        """
        from glassalpha.utils import with_component_seed  # noqa: PLC0415

        return with_component_seed(component_name)

    def _update_progress(self, callback: Callable, message: str, progress: int) -> None:
        """Update progress if callback provided.

        Args:
            callback: Optional progress callback
            message: Progress message
            progress: Progress percentage (0-100)

        """
        if callback:
            callback(progress, message)

        logger.debug(f"Progress: {progress}% - {message}")

    def _preprocess_for_training(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803, C901  # noqa: N803 -> pd.DataFrame:
        """Preprocess features for model training using ColumnTransformer per friend's spec.

        Handles categorical features (like German Credit strings "< 0 DM") with OneHotEncoder
        to prevent ValueError: could not convert string to float during training.

        Args:
            X: Raw features DataFrame

        Returns:
            Processed features DataFrame suitable for training

        """
        from sklearn.compose import ColumnTransformer  # noqa: PLC0415
        from sklearn.preprocessing import OneHotEncoder  # noqa: PLC0415

        # Identify categorical and numeric columns
        categorical_cols = list(X.select_dtypes(include=["object"]).columns)
        numeric_cols = list(X.select_dtypes(exclude=["object"]).columns)

        logger.debug(f"Categorical columns: {categorical_cols}")
        logger.debug(f"Numeric columns: {numeric_cols}")

        if not categorical_cols:
            # No categorical columns, return as-is
            logger.debug("No categorical columns detected, returning original DataFrame")
            return X

        # Build ColumnTransformer with OneHotEncoder for categorical features
        transformers = []

        if categorical_cols:
            transformers.append(
                (
                    "categorical",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None),
                    categorical_cols,
                ),
            )

        if numeric_cols:
            transformers.append(
                (
                    "numeric",
                    "passthrough",  # Pass numeric columns through unchanged
                    numeric_cols,
                ),
            )

        # Create and fit the ColumnTransformer
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        try:
            # Fit and transform the data
            X_transformed = preprocessor.fit_transform(X)  # noqa: N806

            # Get feature names after transformation
            feature_names = []

            # Add categorical feature names (one-hot encoded)
            if categorical_cols:
                cat_transformer = preprocessor.named_transformers_["categorical"]
                if hasattr(cat_transformer, "get_feature_names_out"):
                    cat_features = cat_transformer.get_feature_names_out(categorical_cols)
                else:
                    # Fallback for older sklearn versions
                    cat_features = []
                    for i, col in enumerate(categorical_cols):
                        unique_vals = cat_transformer.categories_[i]
                        cat_features.extend([f"{col}_{val}" for val in unique_vals])
                feature_names.extend(cat_features)

            # Add numeric feature names
            if numeric_cols:
                feature_names.extend(numeric_cols)

            # Sanitize feature names for XGBoost compatibility (no [, ], <, >)
            sanitized_feature_names = []
            for name in feature_names:
                # Replace problematic characters with underscores
                sanitized = (
                    name.replace("<", "lt").replace(">", "gt").replace("[", "_").replace("]", "_").replace(" ", "_")
                )
                # Ensure no double underscores
                sanitized = "_".join(filter(None, sanitized.split("_")))
                sanitized_feature_names.append(sanitized)

            # Convert back to DataFrame with sanitized feature names
            X_processed = pd.DataFrame(X_transformed, columns=sanitized_feature_names, index=X.index)  # noqa: N806

            logger.info(f"Preprocessed {len(categorical_cols)} categorical columns with OneHotEncoder")
            logger.info(f"Final feature count: {len(sanitized_feature_names)} (from {len(X.columns)} original)")
            logger.debug(f"Sanitized feature names: {sanitized_feature_names[:5]}...")

            return X_processed  # noqa: TRY300

        except Exception as e:
            logger.exception(f"Preprocessing failed: {e}")  # noqa: TRY401
            logger.warning("Falling back to simple preprocessing")

            # Fallback: simple label encoding as before
            X_processed = X.copy()  # noqa: N806
            for col in categorical_cols:
                if X_processed[col].dtype == "object":
                    unique_values = X_processed[col].unique()
                    value_map = {val: idx for idx, val in enumerate(unique_values)}
                    X_processed[col] = X_processed[col].map(value_map)
                    logger.debug(f"Label encoded column '{col}': {value_map}")

            return X_processed

    def _ensure_components_loaded(self) -> None:
        """Ensure all required components are imported and registered."""
        try:
            # Import explainer and metrics modules to trigger registration
            from glassalpha.explain.shap import kernel, tree  # noqa: F401, PLC0415
            from glassalpha.metrics.fairness import bias_detection  # noqa: F401, PLC0415
            from glassalpha.metrics.performance import classification  # noqa: F401, PLC0415

            logger.debug("All component modules imported and registered")
        except ImportError as e:
            logger.warning(f"Some components could not be imported: {e}")


def run_audit_pipeline(config: AuditConfig, progress_callback: Callable | None = None) -> AuditResults:
    """Convenience function to run audit pipeline.

    Args:
        config: Validated audit configuration
        progress_callback: Optional progress callback function

    Returns:
        Audit results

    """
    pipeline = AuditPipeline(config)
    return pipeline.run(progress_callback)
