"""Audit manifest generation for complete audit trail.

This module creates comprehensive audit manifests that capture every aspect
of the audit process for regulatory compliance and reproducibility.
"""

import json
import logging
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from glassalpha.constants import make_manifest_component

from .hashing import hash_config, hash_dataframe, hash_file, hash_object
from .seeds import get_seeds_manifest

logger = logging.getLogger(__name__)


class EnvironmentInfo(BaseModel):
    """System and environment information."""

    python_version: str
    platform: str
    architecture: str
    processor: str
    hostname: str
    user: str
    working_directory: str
    environment_variables: dict[str, str] = Field(default_factory=dict)


class GitInfo(BaseModel):
    """Git repository information."""

    commit_hash: str | None = None
    branch: str | None = None
    is_dirty: bool = False
    remote_url: str | None = None
    commit_message: str | None = None
    commit_timestamp: str | None = None


class ComponentInfo(BaseModel):
    """Information about selected components."""

    name: str
    type: str  # 'model', 'explainer', 'metric'
    version: str | None = None
    priority: int | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class DataInfo(BaseModel):
    """Information about datasets used."""

    path: str | None = None
    hash: str | None = None
    shape: tuple[int, int] | None = None
    columns: list[str] = Field(default_factory=list)
    missing_values: dict[str, int] = Field(default_factory=dict)
    target_column: str | None = None
    sensitive_features: list[str] = Field(default_factory=list)


class ExecutionInfo(BaseModel):
    """Information about execution."""

    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    duration_seconds: float | None = None
    status: str = "pending"  # pending, running, completed, failed - tests expect "pending" default
    error_message: str | None = None


class ManifestComponent(BaseModel):
    """Component information for manifest."""

    name: str
    type: str
    details: dict[str, Any] = Field(default_factory=dict)


class AuditManifest(BaseModel):
    """Complete audit manifest with all lineage information."""

    # Basic metadata
    manifest_version: str = "1.0"
    audit_id: str
    version: str = "1.0.0"  # Tests expect this field to exist
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())  # String for test compatibility
    creation_time: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Configuration
    config: dict[str, Any] = Field(default_factory=dict)
    config_hash: str | None = None
    data_hash: str | None = None  # Root-level data hash for e2e tests
    audit_profile: str | None = None
    strict_mode: bool = False

    # Root-level status fields (tests expect these)
    status: str = "pending"
    error_message: str | None = None

    # Environment with defaults (tests expect this to be populated)
    environment: EnvironmentInfo = Field(
        default_factory=lambda: EnvironmentInfo(
            python_version=platform.python_version(),
            platform=platform.platform(),
            architecture=platform.architecture()[0],
            processor=platform.processor() or "unknown",
            hostname=platform.node(),
            user=Path.home().name,
            working_directory=str(Path.cwd()),
            environment_variables={},
            installed_packages={},
        ),
    )
    execution: ExecutionInfo = Field(default_factory=ExecutionInfo)
    execution_info: ExecutionInfo = Field(default_factory=ExecutionInfo)  # Alias for tests
    git: GitInfo | None = None

    # Seeds and reproducibility
    seeds: dict[str, Any] = Field(default_factory=dict)
    deterministic_validation: dict[str, bool | None] = Field(default_factory=dict)

    # Components (tests expect this field name)
    components: dict[str, dict[str, Any]] = Field(default_factory=dict)  # Changed to dict for test compatibility
    selected_components: dict[str, dict[str, Any]] = Field(default_factory=dict)  # Support nested structure

    # Data
    datasets: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
    )  # Changed to dict for test compatibility with 'name' field

    # Execution (made optional for minimal construction)

    # Results hashes
    result_hashes: dict[str, str] = Field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        """Export manifest as JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation

        """
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict[str, Any]:
        """Export manifest as dictionary.

        Returns:
            Dictionary representation

        """
        return self.model_dump()

    def save(self, path: Path) -> None:
        """Save audit manifest to JSON file for regulatory compliance and reproducibility.

        Creates a permanent record of the audit execution including all component
        selections, data hashes, environment details, and configuration parameters.
        This manifest enables full audit trail documentation required for regulatory
        submissions and reproducible audit verification.

        Args:
            path: Target file path for manifest storage (typically .json extension)

        Side Effects:
            - Creates or overwrites JSON file at specified path
            - File contains sensitive environment information (review before sharing)
            - Timestamps record exact audit execution time in UTC
            - File size typically 5-50KB depending on configuration complexity

        Raises:
            IOError: If path is not writable or insufficient disk space
            JSONEncodeError: If manifest contains non-serializable data

        Note:
            This manifest is required for regulatory audit trails and must be
            preserved with the corresponding audit report for compliance verification.

        """
        path = Path(path)

        with path.open("w") as f:
            f.write(self.to_json())

        logger.info("Audit manifest saved to %s", path)


class ManifestGenerator:
    """Generator for comprehensive audit manifests."""

    def __init__(self, audit_id: str | None = None) -> None:
        """Initialize manifest generator.

        Args:
            audit_id: Unique audit identifier (auto-generated if None)

        """
        self.audit_id = audit_id or self._generate_audit_id()
        self.start_time = datetime.now(UTC)

        # Direct attributes for test compatibility
        self.status: str = "initialized"
        self.error: str | None = None
        self.completed_at: datetime | None = None

        # Attributes tests expect to exist
        self.seeds: dict[str, Any] | None = None
        self.config: dict[str, Any] | None = None
        self.config_hash: str | None = None
        self.components: dict[str, ComponentInfo] = {}
        self.datasets: dict[str, Any] = {}
        self.result_hashes: dict[str, str] = {}

        # Initialize empty manifest (don't collect platform/git on init to avoid crashes)
        self.manifest = AuditManifest(
            audit_id=self.audit_id,
        )

        # Sync start_time between execution and execution_info fields
        self.manifest.execution.start_time = self.start_time
        self.manifest.execution_info.start_time = self.start_time

        logger.info("Initialized audit manifest: %s", self.audit_id)

    def add_config(self, config: dict[str, Any]) -> None:
        """Add configuration to manifest.

        Args:
            config: Configuration dictionary

        """
        # Update direct attributes for test compatibility
        self.config = config
        self.config_hash = hash_config(config)

        # Update manifest
        self.manifest.config = config
        self.manifest.config_hash = self.config_hash
        self.manifest.audit_profile = config.get("audit_profile")
        self.manifest.strict_mode = config.get("strict_mode", False)

        logger.debug("Added configuration to manifest")

    def add_seeds(self) -> None:
        """Add seed information to manifest."""
        from .seeds import validate_deterministic_environment  # noqa: PLC0415

        # Update direct attribute for test compatibility
        seeds_data = get_seeds_manifest()
        self.seeds = seeds_data

        # Update manifest
        self.manifest.seeds = seeds_data
        self.manifest.deterministic_validation = validate_deterministic_environment()

        logger.debug("Added seed information to manifest")

    def add_component(
        self,
        name: str,
        implementation: str,
        obj: Any = None,  # noqa: ANN401
        *,
        config: Any = None,  # noqa: ANN401
        priority: Any = None,  # noqa: ANN401
    ) -> None:
        """Add component to manifest with friend's spec for test compatibility.

        Args:
            name: Component role (e.g. 'model', 'explainer')
            implementation: Component implementation (e.g. "xgboost", "treeshap")
            obj: Optional component object for version extraction
            config: Component configuration
            priority: Component priority

        """
        # components: detailed catalog
        self.manifest.components[name] = {
            "name": name,
            "type": implementation,
            "details": {
                "implementation": implementation,
                "version": getattr(obj, "version", "1.0.0"),
                **({"priority": priority} if priority is not None else {}),
                **({"config": config} if config is not None else {}),
            },
        }

        # selected_components: compact summary the tests inspect
        # Uses centralized helper to ensure exact format E2E tests expect
        self.manifest.selected_components[name] = make_manifest_component(name, implementation)

        # Store in generator components for backward compatibility
        if hasattr(self, "components"):
            component = ManifestComponent(
                name=name,
                type=implementation,
                details={"implementation": implementation, "version": getattr(obj, "version", "1.0.0")},
            )
            self.components[name] = component

        logger.debug("Added component to manifest: %s (%s)", name, implementation)

    def add_dataset(
        self,
        dataset_name: str,
        file_path: Path | None = None,
        data: pd.DataFrame | None = None,
        target_column: str | None = None,
        sensitive_features: list[str] | None = None,
    ) -> None:
        """Add dataset information to manifest.

        Args:
            dataset_name: Name of dataset
            data: DataFrame (for shape/hash info)
            file_path: Original file path
            target_column: Target column name
            sensitive_features: List of sensitive feature names

        """
        dataset_info = DataInfo(
            path=str(file_path) if file_path is not None else None,
            target_column=target_column,
            sensitive_features=sensitive_features or [],
        )

        # Add data information if available - avoid DataFrame truth ambiguity
        if data is not None and hasattr(data, "shape"):  # Check for shape attribute instead of truthiness
            dataset_info.hash = hash_dataframe(data)
            # Also set root-level data_hash for e2e tests (use the main dataset's hash)
            if dataset_name == "main" or self.manifest.data_hash is None:
                self.manifest.data_hash = dataset_info.hash
            dataset_info.shape = data.shape
            dataset_info.columns = list(data.columns)
            dataset_info.missing_values = data.isna().sum().to_dict()
        elif file_path is not None and file_path.exists():  # Explicit None check
            dataset_info.hash = hash_file(file_path)
            # Also set root-level data_hash for e2e tests (use the main dataset's hash)
            if dataset_name == "main" or self.manifest.data_hash is None:
                self.manifest.data_hash = dataset_info.hash

        # Update both manifest and direct attribute for test compatibility
        # Ensure datasets include 'name' field as expected by tests
        dataset_dict = dataset_info.model_dump() if hasattr(dataset_info, "model_dump") else vars(dataset_info)
        dataset_dict["name"] = dataset_name  # Add name field for test compatibility

        self.manifest.datasets[dataset_name] = dataset_dict
        self.datasets[dataset_name] = dataset_dict
        logger.debug("Added dataset to manifest: %s", dataset_name)

    def add_result_hash(self, result_name: str, result_hash: str) -> None:
        """Add hash of result/output to manifest.

        Args:
            result_name: Name of result (e.g., 'report', 'explanations')
            result_hash: Hash of result

        """
        # Update both manifest and direct attribute for test compatibility
        self.manifest.result_hashes[result_name] = result_hash
        self.result_hashes[result_name] = result_hash
        logger.debug("Added result hash: %s", result_name)

    def mark_completed(self, status: str = "completed", error: str | None = None) -> None:
        """Mark audit as completed.

        Args:
            status: Final status ('completed', 'failed', 'success')
            error: Error message if failed

        """
        # Validate status
        if status not in {"completed", "failed", "success"}:  # Added "success" for test compatibility
            msg = "status must be 'completed', 'failed', or 'success'"
            raise ValueError(msg)

        end_time = datetime.now(UTC)

        # Update direct attributes for test compatibility
        self.status = status
        self.error = error
        self.completed_at = end_time

        # Keep manifest dict in sync
        self.manifest.execution.end_time = end_time
        self.manifest.execution.duration_seconds = (end_time - self.start_time).total_seconds()
        self.manifest.execution.status = status

        # Also update execution_info alias for test compatibility
        self.manifest.execution_info.end_time = end_time
        self.manifest.execution_info.duration_seconds = (end_time - self.start_time).total_seconds()
        self.manifest.execution_info.status = status

        # Friend's spec: Set root-level status and error_message in manifest
        self.manifest.status = status
        self.manifest.error_message = error

        if error:
            self.manifest.execution.error_message = error
            self.manifest.execution_info.error_message = error

        logger.info("Audit marked as %s (duration: %.2fs)", status, self.manifest.execution.duration_seconds)

    def finalize(self) -> AuditManifest:
        """Finalize and return the completed manifest.

        Returns:
            Complete audit manifest

        """
        if self.manifest.execution.status == "running":
            self.mark_completed()

        return self.manifest

    def _generate_audit_id(self) -> str:
        """Generate unique audit ID.

        Returns:
            Unique audit ID string

        """
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        hash_input = f"{timestamp}_{platform.node()}_{os.getpid()}"
        audit_hash = hash_object(hash_input)[:8]

        return f"audit_{timestamp}_{audit_hash}"

    def _collect_environment_info(self) -> EnvironmentInfo:
        """Collect system and environment information.

        Returns:
            Environment information

        """
        # Collect key environment variables (excluding sensitive ones)
        sensitive_keys = {"PASSWORD", "TOKEN", "SECRET", "KEY", "CREDENTIAL"}

        # Use dictionary comprehension for better performance
        safe_env_vars = {
            key: value
            for key, value in os.environ.items()
            if (key.startswith(("PYTHON", "GLASSALPHA", "PATH")) or key in {"USER", "HOME", "PWD"})
            and not any(sensitive in key.upper() for sensitive in sensitive_keys)
        }

        return EnvironmentInfo(
            python_version=sys.version,
            platform=platform.platform(),
            architecture=platform.architecture()[0],
            processor=platform.processor(),
            hostname=platform.node(),
            user=os.environ.get("USER", "unknown"),
            working_directory=str(Path.cwd()),
            environment_variables=safe_env_vars,
        )

    def _collect_git_info(self) -> GitInfo | None:
        """Collect Git repository information.

        Returns:
            Git information if available, None otherwise

        """

        def _run(*args: str) -> str:
            """Helper to run git commands safely."""
            p = subprocess.run(args, capture_output=True, check=False, text=True)  # noqa: S603
            return (p.stdout or "").strip()

        # Friend's spec: use subprocess.run with text=True and never .decode()
        info = {
            "commit_sha": _run("git", "rev-parse", "HEAD"),
            "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "remote_url": _run("git", "config", "--get", "remote.origin.url"),
            "last_commit_message": _run("git", "log", "-1", "--pretty=%B"),
            "last_commit_date": _run("git", "log", "-1", "--date=iso-strict", "--pretty=%cd"),
        }
        info["is_dirty"] = bool(_run("git", "status", "--porcelain"))

        return GitInfo(
            commit_hash=info["commit_sha"],
            branch=info["branch"],
            is_dirty=info["is_dirty"],
            remote_url=info["remote_url"] or None,
            commit_message=info["last_commit_message"],
            commit_timestamp=info["last_commit_date"],
        )


def create_manifest(audit_id: str | None = None) -> ManifestGenerator:
    """Create a new audit manifest generator.

    Args:
        audit_id: Optional audit ID (auto-generated if None)

    Returns:
        Manifest generator instance

    """
    return ManifestGenerator(audit_id)


def load_manifest(path: Path) -> AuditManifest:
    """Load existing audit manifest from file.

    Args:
        path: Path to manifest file

    Returns:
        Loaded audit manifest

    Raises:
        FileNotFoundError: If manifest file doesn't exist
        ValueError: If manifest is invalid

    """
    path = Path(path)

    if not path.exists():
        msg = f"Manifest file not found: {path}"
        raise FileNotFoundError(msg)

    try:
        with path.open() as f:
            manifest_data = json.load(f)

        return AuditManifest(**manifest_data)

    except Exception as e:
        msg = f"Invalid manifest file {path}: {e}"
        raise ValueError(msg) from e


def compare_manifests(manifest1: AuditManifest, manifest2: AuditManifest) -> dict[str, Any]:
    """Compare two audit manifests for differences.

    Args:
        manifest1: First manifest
        manifest2: Second manifest

    Returns:
        Dictionary with comparison results

    """
    comparison = {
        "identical": True,
        "differences": [],
        "config_hash_match": manifest1.config_hash == manifest2.config_hash,
        "seed_match": manifest1.seeds == manifest2.seeds,
        "components_match": manifest1.selected_components == manifest2.selected_components,
        "data_hash_match": True,
    }

    # Compare data hashes
    for dataset_name, dataset1 in manifest1.datasets.items():
        if dataset_name in manifest2.datasets:
            dataset2 = manifest2.datasets[dataset_name]
            if dataset1.hash != dataset2.hash:
                comparison["data_hash_match"] = False
                comparison["differences"].append(f"Data hash mismatch for {dataset_name}")
        else:
            comparison["differences"].append(f"Dataset {dataset_name} missing in second manifest")

    # Overall comparison
    if not all(
        [
            comparison["config_hash_match"],
            comparison["seed_match"],
            comparison["components_match"],
            comparison["data_hash_match"],
        ],
    ):
        comparison["identical"] = False

    return comparison
