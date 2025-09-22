"""Audit profile system for Glass Alpha.

Profiles define valid component combinations for different audit types,
ensuring appropriate metrics, explainers, and report templates are used.
"""

from .base import BaseAuditProfile
from .tabular import TabularComplianceProfile

from ..core.registry import ProfileRegistry

# Auto-register profiles on import
ProfileRegistry.register("tabular_compliance")(TabularComplianceProfile)

__all__ = [
    "BaseAuditProfile",
    "TabularComplianceProfile",
]
