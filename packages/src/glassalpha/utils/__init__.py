"""Core utilities for reproducibility and audit tracking.

This package provides essential utilities for ensuring deterministic
behavior across the entire audit pipeline:
- Centralized seed management
- Deterministic hashing
- Comprehensive audit manifests
"""

from .hashing import hash_config, hash_dataframe, hash_file, hash_object
from .manifest import AuditManifest, ManifestGenerator
from .seeds import (
    SeedManager,
    get_component_seed,
    get_seeds_manifest,
    set_global_seed,
    with_component_seed,
    with_seed,
)

__all__ = [
    "SeedManager",
    "get_component_seed",
    "get_seeds_manifest",
    "set_global_seed",
    "with_component_seed",
    "with_seed",
    "hash_config",
    "hash_dataframe",
    "hash_file",
    "hash_object",
    "AuditManifest",
    "ManifestGenerator",
]
