"""Audit profile system for GlassAlpha.

Profiles define valid component combinations for different audit types,
ensuring appropriate metrics, explainers, and report templates are used.
"""

from ..core.registry import ProfileRegistry
from .base import BaseAuditProfile
from .tabular import TabularComplianceProfile

# Auto-register profiles on import
ProfileRegistry.register("tabular_compliance")(TabularComplianceProfile)

__all__ = [
    "BaseAuditProfile",
    "TabularComplianceProfile",
]
