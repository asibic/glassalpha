"""Tabular compliance audit profile for Phase 1.

This profile defines the component set for regulatory compliance audits
of tabular machine learning models (XGBoost, LightGBM, LogisticRegression).
"""

from typing import Any

from .base import BaseAuditProfile


class TabularComplianceProfile(BaseAuditProfile):
    """Profile for tabular model compliance audits.

    This is the primary profile for Phase 1, focusing on tree-based models
    and their regulatory compliance requirements.
    """

    name = "tabular_compliance"
    version = "1.0.0"

    # Compatible models for Phase 1
    compatible_models = [
        "xgboost",
        "lightgbm",
        "logistic_regression",
        "random_forest",
        "passthrough",  # For testing
    ]

    # Required metrics for compliance
    required_metrics = [
        # Performance metrics
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc_roc",
        # Fairness metrics
        "demographic_parity",
        "equal_opportunity",
        "statistical_parity",
        # Drift metrics
        "psi",  # Population Stability Index
        "kl_divergence",
    ]

    # Optional metrics that can be included
    optional_metrics = [
        "confusion_matrix",
        "calibration",
        "feature_importance",
        "partial_dependence",
        "noop",  # For testing
    ]

    # Report template for tabular audits
    report_template = "standard_audit"

    # Explainer priority for tabular models
    explainer_priority = [
        "treeshap",  # Best for tree models
        "kernelshap",  # Fallback for any model
        "noop",  # Last resort for testing
    ]

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> bool:
        """Validate configuration for tabular compliance.

        Args:
            config: Configuration to validate

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid

        """
        # Run parent validation
        super().validate_config(config)

        # Additional tabular-specific validation
        data_config = config.get("data", {})

        # Check for protected attributes (needed for fairness)
        if "protected_attributes" not in data_config:
            raise ValueError(
                f"Profile '{cls.name}' requires 'protected_attributes' in data configuration for fairness analysis"
            )

        # Check for schema (needed for determinism)
        if "schema_path" not in data_config and "schema" not in data_config:
            raise ValueError(f"Profile '{cls.name}' requires data schema for deterministic validation")

        # Check explainer configuration
        explainer_config = config.get("explainers", {})
        if not explainer_config.get("priority"):
            # Provide default if not specified
            config["explainers"] = config.get("explainers", {})
            config["explainers"]["priority"] = cls.explainer_priority

        # Check metrics configuration
        metrics_config = config.get("metrics", {})
        if not metrics_config:
            # Set default metrics if not specified
            config["metrics"] = {
                "performance": ["accuracy", "precision", "recall", "f1", "auc_roc"],
                "fairness": ["demographic_parity", "equal_opportunity"],
                "drift": ["psi"],
            }

        # Validate recourse configuration if present
        if "recourse" in config:
            recourse_config = config["recourse"]
            if recourse_config.get("enabled", False) and "immutable_features" not in recourse_config:
                raise ValueError("Recourse requires 'immutable_features' to be specified")

        return True

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Get default configuration for this profile.

        Returns:
            Default configuration dictionary

        """
        return {
            "audit_profile": cls.name,
            "model": {
                "type": "xgboost",  # Most common
                "path": None,  # Must be provided
            },
            "data": {
                "path": None,  # Must be provided
                "schema_path": None,  # Must be provided
                "protected_attributes": [],  # Must be provided
            },
            "explainers": {
                "strategy": "first_compatible",
                "priority": cls.explainer_priority,
                "config": {
                    "treeshap": {
                        "max_samples": 1000,
                        "check_additivity": True,
                    },
                    "kernelshap": {
                        "n_samples": 100,
                    },
                },
            },
            "metrics": {
                "performance": ["accuracy", "precision", "recall", "f1", "auc_roc"],
                "fairness": ["demographic_parity", "equal_opportunity"],
                "drift": ["psi"],
            },
            "recourse": {
                "enabled": True,
                "immutable_features": [],
                "monotonic_constraints": {},
                "cost_function": "weighted_l1",
            },
            "report": {
                "template": cls.report_template,
                "output_format": "pdf",
                "include_sections": [
                    "lineage",
                    "data_schema",
                    "global_explanations",
                    "local_explanations",
                    "fairness",
                    "drift",
                    "recourse",
                    "assumptions",
                ],
            },
            "reproducibility": {
                "random_seed": 42,
                "deterministic": True,
                "capture_environment": True,
            },
            "manifest": {
                "enabled": True,
                "include_git_sha": True,
                "include_config_hash": True,
                "include_data_hash": True,
            },
            "strict_mode": False,
        }
