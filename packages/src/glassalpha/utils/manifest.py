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

    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: float | None = None
    status: str = "running"  # running, completed, failed
    error_message: str | None = None


class AuditManifest(BaseModel):
    """Complete audit manifest with all lineage information."""

    # Basic metadata
    manifest_version: str = "1.0"
    audit_id: str
    creation_time: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Configuration
    config: dict[str, Any] = Field(default_factory=dict)
    config_hash: str | None = None
    audit_profile: str | None = None
    strict_mode: bool = False

    # Environment
    environment: EnvironmentInfo
    git: GitInfo | None = None

    # Seeds and reproducibility
    seeds: dict[str, Any] = Field(default_factory=dict)
    deterministic_validation: dict[str, bool | None] = Field(default_factory=dict)

    # Components
    selected_components: dict[str, ComponentInfo] = Field(default_factory=dict)

    # Data
    datasets: dict[str, DataInfo] = Field(default_factory=dict)

    # Execution
    execution: ExecutionInfo

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

        with open(path, "w") as f:
            f.write(self.to_json())

        logger.info(f"Audit manifest saved to {path}")


class ManifestGenerator:
    """Generator for comprehensive audit manifests."""

    def __init__(self, audit_id: str | None = None):
        """Initialize manifest generator.

        Args:
            audit_id: Unique audit identifier (auto-generated if None)

        """
        self.audit_id = audit_id or self._generate_audit_id()
        self.start_time = datetime.now(UTC)

        # Initialize manifest with basic info
        self.manifest = AuditManifest(
            audit_id=self.audit_id,
            environment=self._collect_environment_info(),
            git=self._collect_git_info(),
            execution=ExecutionInfo(start_time=self.start_time),
        )

        logger.info(f"Initialized audit manifest: {self.audit_id}")

    def add_config(self, config: dict[str, Any]) -> None:
        """Add configuration to manifest.

        Args:
            config: Configuration dictionary

        """
        self.manifest.config = config
        self.manifest.config_hash = hash_config(config)
        self.manifest.audit_profile = config.get("audit_profile")
        self.manifest.strict_mode = config.get("strict_mode", False)

        logger.debug("Added configuration to manifest")

    def add_seeds(self) -> None:
        """Add seed information to manifest."""
        from .seeds import validate_deterministic_environment

        self.manifest.seeds = get_seeds_manifest()
        self.manifest.deterministic_validation = validate_deterministic_environment()

        logger.debug("Added seed information to manifest")

    def add_component(
        self,
        component_type: str,
        component_name: str,
        component: Any,
        priority: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Add selected component to manifest.

        Args:
            component_type: Type of component ('model', 'explainer', 'metric')
            component_name: Name/ID of component
            component: Component instance
            priority: Selection priority
            config: Component configuration

        """
        # Extract component information
        component_info = ComponentInfo(name=component_name, type=component_type, priority=priority, config=config or {})

        # Try to get version and capabilities
        if hasattr(component, "version"):
            component_info.version = component.version

        if hasattr(component, "capabilities"):
            component_info.capabilities = component.capabilities
        elif hasattr(component, "get_capabilities"):
            component_info.capabilities = component.get_capabilities()

        # Store in manifest
        key = f"{component_type}_{component_name}"
        self.manifest.selected_components[key] = component_info

        logger.debug(f"Added component to manifest: {key}")

    def add_dataset(
        self,
        dataset_name: str,
        data: pd.DataFrame | None = None,
        file_path: Path | None = None,
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
            path=str(file_path) if file_path else None,
            target_column=target_column,
            sensitive_features=sensitive_features or [],
        )

        # Add data information if available
        if data is not None:
            dataset_info.hash = hash_dataframe(data)
            dataset_info.shape = data.shape
            dataset_info.columns = list(data.columns)
            dataset_info.missing_values = data.isnull().sum().to_dict()
        elif file_path and file_path.exists():
            dataset_info.hash = hash_file(file_path)

        self.manifest.datasets[dataset_name] = dataset_info
        logger.debug(f"Added dataset to manifest: {dataset_name}")

    def add_result_hash(self, result_name: str, result_hash: str) -> None:
        """Add hash of result/output to manifest.

        Args:
            result_name: Name of result (e.g., 'report', 'explanations')
            result_hash: Hash of result

        """
        self.manifest.result_hashes[result_name] = result_hash
        logger.debug(f"Added result hash: {result_name}")

    def mark_completed(self, status: str = "completed", error: str | None = None) -> None:
        """Mark audit as completed.

        Args:
            status: Final status ('completed', 'failed')
            error: Error message if failed

        """
        end_time = datetime.now(UTC)

        self.manifest.execution.end_time = end_time
        self.manifest.execution.duration_seconds = (end_time - self.start_time).total_seconds()
        self.manifest.execution.status = status

        if error:
            self.manifest.execution.error_message = error

        logger.info(f"Audit marked as {status} (duration: {self.manifest.execution.duration_seconds:.2f}s)")

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
        safe_env_vars = {}
        sensitive_keys = {"PASSWORD", "TOKEN", "SECRET", "KEY", "CREDENTIAL"}

        for key, value in os.environ.items():
            # Only include relevant non-sensitive variables
            if (key.startswith(("PYTHON", "GLASSALPHA", "PATH")) or key in {"USER", "HOME", "PWD"}) and not any(
                sensitive in key.upper() for sensitive in sensitive_keys
            ):
                safe_env_vars[key] = value

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
        try:
            # Check if we're in a git repository
            subprocess.run(["git", "status"], capture_output=True, check=True, cwd=Path.cwd())

            # Get commit hash
            commit_result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
            commit_hash = commit_result.stdout.strip()

            # Get branch name
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"], capture_output=True, text=True, check=True
            )
            branch = branch_result.stdout.strip()

            # Check if working directory is dirty
            status_result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
            is_dirty = bool(status_result.stdout.strip())

            # Get remote URL
            try:
                remote_result = subprocess.run(
                    ["git", "remote", "get-url", "origin"], capture_output=True, text=True, check=True
                )
                remote_url = remote_result.stdout.strip()
            except subprocess.CalledProcessError:
                remote_url = None

            # Get commit message and timestamp
            try:
                commit_info_result = subprocess.run(
                    ["git", "log", "-1", "--pretty=format:%s|%ci"], capture_output=True, text=True, check=True
                )
                commit_info = commit_info_result.stdout.strip().split("|")
                commit_message = commit_info[0] if len(commit_info) > 0 else None
                commit_timestamp = commit_info[1] if len(commit_info) > 1 else None
            except subprocess.CalledProcessError:
                commit_message = None
                commit_timestamp = None

            return GitInfo(
                commit_hash=commit_hash,
                branch=branch,
                is_dirty=is_dirty,
                remote_url=remote_url,
                commit_message=commit_message,
                commit_timestamp=commit_timestamp,
            )

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Not in a git repository or git not available
            logger.debug("Git information not available")
            return None


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
        raise FileNotFoundError(f"Manifest file not found: {path}")

    try:
        with open(path) as f:
            manifest_data = json.load(f)

        return AuditManifest(**manifest_data)

    except Exception as e:
        raise ValueError(f"Invalid manifest file {path}: {e}") from e


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
        ]
    ):
        comparison["identical"] = False

    return comparison
