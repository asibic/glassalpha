"""Configuration schema definitions using Pydantic.

This module defines the structure and validation for GlassAlpha
configuration files, ensuring type safety and consistency.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CalibrationConfig(BaseModel):
    """Probability calibration configuration."""

    model_config = ConfigDict(extra="forbid")

    method: str | None = Field(None, description="Calibration method ('isotonic' or 'sigmoid')")
    cv: int = Field(5, description="Number of cross-validation folds")
    ensemble: bool = Field(True, description="Whether to use ensemble calibration")

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str | None) -> str | None:
        """Validate calibration method."""
        if v is None:
            return v
        v = v.lower()
        if v not in {"isotonic", "sigmoid"}:
            raise ValueError(f"Calibration method must be 'isotonic' or 'sigmoid', got: {v}")
        return v

    @field_validator("cv")
    @classmethod
    def validate_cv(cls, v: int) -> int:
        """Validate CV folds."""
        if v < 2:
            raise ValueError(f"CV folds must be >= 2, got: {v}")
        return v


class ModelConfig(BaseModel):
    """Model configuration."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., description="Model type (xgboost, lightgbm, logistic_regression, etc.)")
    path: Path | None = Field(None, description="Path to saved model file")
    params: dict[str, Any] | None = Field(default_factory=dict, description="Additional model parameters")
    calibration: CalibrationConfig | None = Field(None, description="Optional probability calibration")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Ensure model type is lowercase."""
        return v.lower()


class DataConfig(BaseModel):
    """Data configuration."""

    model_config = ConfigDict(extra="forbid")

    path: Path | None = Field(None, description="Path to data file")
    schema_path: Path | None = Field(None, description="Path to data schema file")
    schema: dict[str, Any] | None = Field(None, description="Inline data schema")
    protected_attributes: list[str] = Field(
        default_factory=list,
        description="List of protected/sensitive attributes for fairness analysis",
    )
    target_column: str | None = Field(None, description="Name of target column")
    feature_columns: list[str] | None = Field(None, description="List of feature columns to use")

    @field_validator("protected_attributes")
    @classmethod
    def lowercase_attributes(cls, v: list[str]) -> list[str]:
        """Ensure attribute names are lowercase."""
        return [attr.lower() for attr in v]


class ExplainerConfig(BaseModel):
    """Explainer configuration."""

    model_config = ConfigDict(extra="forbid")

    strategy: str = Field("first_compatible", description="Selection strategy (first_compatible, best_score)")
    priority: list[str] = Field(default_factory=list, description="Ordered list of explainer preferences")
    config: dict[str, dict[str, Any]] = Field(default_factory=dict, description="Per-explainer configuration")
    enabled: bool = Field(True, description="Whether to generate explanations")


class MetricCategory(BaseModel):
    """Configuration for a category of metrics."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(True, description="Whether to compute these metrics")
    metrics: list[str] = Field(default_factory=list, description="Metric names")
    config: dict[str, Any] = Field(default_factory=dict, description="Category-specific configuration")


class MetricsConfig(BaseModel):
    """Metrics configuration."""

    model_config = ConfigDict(extra="forbid")

    performance: list[str] | MetricCategory = Field(
        default_factory=lambda: MetricCategory(metrics=["accuracy", "precision", "recall", "f1", "auc_roc"]),
        description="Performance metrics",
    )
    fairness: list[str] | MetricCategory = Field(
        default_factory=lambda: MetricCategory(metrics=["demographic_parity", "equal_opportunity"]),
        description="Fairness metrics",
    )
    drift: list[str] | MetricCategory = Field(
        default_factory=lambda: MetricCategory(metrics=["psi"]),
        description="Drift metrics",
    )
    custom: dict[str, list[str]] | None = Field(None, description="Custom metric categories")

    @field_validator("performance", "fairness", "drift")
    @classmethod
    def convert_list_to_category(cls, v):
        """Convert list format to MetricCategory."""
        if isinstance(v, list):
            return MetricCategory(metrics=v)
        return v


class RecourseConfig(BaseModel):
    """Recourse configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(False, description="Whether to generate recourse")
    immutable_features: list[str] = Field(default_factory=list, description="Features that cannot be changed")
    monotonic_constraints: dict[str, str] = Field(
        default_factory=dict,
        description="Monotonic constraints (increase_only, decrease_only)",
    )
    cost_function: str = Field("weighted_l1", description="Cost function for optimization")
    max_iterations: int = Field(100, description="Maximum optimization iterations")

    @field_validator("monotonic_constraints")
    @classmethod
    def validate_constraints(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate constraint values."""
        valid_constraints = {"increase_only", "decrease_only", "fixed"}
        for feature, constraint in v.items():
            if constraint not in valid_constraints:
                raise ValueError(
                    f"Invalid constraint '{constraint}' for feature '{feature}'. Must be one of {valid_constraints}",
                )
        return v


class ReportConfig(BaseModel):
    """Report generation configuration."""

    model_config = ConfigDict(extra="ignore")  # Allow unknown fields like styling

    template: str = Field("standard_audit", description="Report template name")
    output_format: str = Field("pdf", description="Output format (pdf, html, json)")
    include_sections: list[str] = Field(
        default_factory=lambda: [
            "lineage",
            "data_schema",
            "global_explanations",
            "local_explanations",
            "fairness",
            "drift",
            "recourse",
            "assumptions",
        ],
        description="Report sections to include",
    )
    custom_branding: dict[str, Any] | None = Field(None, description="Custom branding configuration (enterprise only)")


class ReproducibilityConfig(BaseModel):
    """Reproducibility configuration."""

    model_config = ConfigDict(extra="forbid")

    random_seed: int | None = Field(42, description="Random seed for determinism")
    deterministic: bool = Field(True, description="Enforce deterministic behavior")
    capture_environment: bool = Field(True, description="Capture environment information")

    @field_validator("random_seed")
    @classmethod
    def validate_seed(cls, v: int | None) -> int | None:
        """Validate seed is positive if provided."""
        if v is not None and v < 0:
            raise ValueError("Random seed must be non-negative")
        return v


class ManifestConfig(BaseModel):
    """Manifest generation configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(True, description="Whether to generate manifest")
    include_git_sha: bool = Field(True, description="Include git commit SHA")
    include_config_hash: bool = Field(True, description="Include configuration hash")
    include_data_hash: bool = Field(True, description="Include data hash")
    include_model_hash: bool = Field(True, description="Include model hash")
    output_path: Path | None = Field(None, description="Path to save manifest (default: alongside report)")


class AuditConfig(BaseModel):
    """Main audit configuration."""

    model_config = ConfigDict(extra="forbid")

    # Required fields
    audit_profile: str = Field(..., description="Audit profile name (e.g., tabular_compliance)")
    model: ModelConfig = Field(..., description="Model configuration")
    data: DataConfig = Field(..., description="Data configuration")

    # Optional fields with defaults
    explainers: ExplainerConfig = Field(default_factory=ExplainerConfig, description="Explainer configuration")
    metrics: MetricsConfig = Field(default_factory=MetricsConfig, description="Metrics configuration")
    recourse: RecourseConfig = Field(default_factory=RecourseConfig, description="Recourse configuration")
    report: ReportConfig = Field(default_factory=ReportConfig, description="Report configuration")
    reproducibility: ReproducibilityConfig = Field(
        default_factory=ReproducibilityConfig,
        description="Reproducibility configuration",
    )
    manifest: ManifestConfig = Field(default_factory=ManifestConfig, description="Manifest configuration")

    # Mode flags
    strict_mode: bool = Field(False, description="Enable strict mode for regulatory compliance")

    # Additional metadata
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata for audit trail")

    @model_validator(mode="before")
    @classmethod
    def _compat_old_fields(cls, values: Any) -> Any:
        """Back-compatibility shim for old flat field names.

        Maps old schema format to new nested structure:
        - data_path -> data.path
        - model_config -> model
        - output -> report
        """
        if isinstance(values, dict):
            # Convert old data_path to new nested data structure
            if "data_path" in values and "data" not in values:
                values["data"] = {"path": values.pop("data_path")}

            # Convert old model_config to new model structure
            if "model_config" in values and "model" not in values:
                values["model"] = values.pop("model_config")

            # Convert old output to new report structure (drop path since it's now CLI-level)
            if "output" in values and "report" not in values:
                old_output = values.pop("output")
                if isinstance(old_output, dict):
                    # Extract report config fields, ignore path (now handled at CLI level)
                    report_config = {k: v for k, v in old_output.items() if k != "path"}
                    if report_config:
                        values["report"] = report_config
                    # Note: output.path is now passed directly to CLI, not in config
        return values

    @field_validator("audit_profile")
    @classmethod
    def validate_profile(cls, v: str) -> str:
        """Ensure profile name is lowercase."""
        return v.lower()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditConfig":
        """Create from dictionary."""
        return cls(**data)
