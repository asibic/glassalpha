# SPDX-License-Identifier: Apache-2.0
"""Constants for fragile contracts - centralized to prevent typos and drift.

These strings are matched exactly by tests and must not change without
updating corresponding test assertions.
"""

# Error messages - matched by tests
NO_MODEL_MSG = "Model not loaded. Load a model first."
MODEL_NOT_FITTED_MSG = "Model not fitted"
NO_MODEL_TO_SAVE_MSG = "No model to save"

# Log messages - matched by pytest assertions
INIT_LOG_TEMPLATE = "Initialized audit pipeline with profile: {profile}"

# Explainer selection errors
NO_COMPATIBLE_EXPLAINER_MSG = "No compatible explainer found"

# File/path errors
FILE_NOT_EXIST_MSG = "does not exist"


# Manifest component structure - matched by E2E tests
def make_manifest_component(name: str, implementation: str) -> dict[str, str]:
    """Create manifest component in exact format E2E tests expect."""
    return {"name": implementation, "type": name}


# Template names - matched by packaging/import tests
STANDARD_AUDIT_TEMPLATE = "standard_audit.html"
