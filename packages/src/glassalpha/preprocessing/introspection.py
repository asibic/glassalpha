"""Introspect sklearn artifacts to extract learned parameters."""

from typing import Any

import numpy as np
import pandas as pd


def extract_sklearn_manifest(artifact: Any) -> dict[str, Any]:
    """Extract learned parameters from sklearn preprocessing artifact.

    Args:
        artifact: Sklearn Pipeline or ColumnTransformer

    Returns:
        Manifest dictionary with components, learned params, versions, etc.

    """
    import sklearn

    from glassalpha.preprocessing.manifest import MANIFEST_SCHEMA_VERSION

    manifest: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "components": [],
        "artifact_runtime_versions": {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "audit_runtime_versions": {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }

    # Add scipy if available
    try:
        import scipy

        manifest["artifact_runtime_versions"]["scipy"] = scipy.__version__
        manifest["audit_runtime_versions"]["scipy"] = scipy.__version__
    except ImportError:
        pass

    # Extract components
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    if isinstance(artifact, Pipeline):
        for name, step in artifact.steps:
            manifest["components"].append(_extract_component(name, step))
    elif isinstance(artifact, ColumnTransformer):
        for name, transformer, columns in artifact.transformers:
            component = _extract_component(name, transformer)
            component["columns"] = list(columns) if columns is not None else None
            manifest["components"].append(component)
    else:
        manifest["components"].append(_extract_component("transformer", artifact))

    # Detect output dtype and sparsity (requires sample transform)
    manifest["output_dtype"] = "unknown"  # Will be set during transform

    return manifest


def _extract_component(name: str, transformer: Any) -> dict[str, Any]:
    """Extract parameters from a single transformer."""
    from sklearn.pipeline import Pipeline

    from glassalpha.preprocessing.validation import fqcn

    component: dict[str, Any] = {
        "name": name,
        "class": fqcn(transformer),
    }

    # Extract type-specific parameters
    class_name = transformer.__class__.__name__

    if class_name == "OneHotEncoder":
        component["handle_unknown"] = transformer.handle_unknown
        component["drop"] = str(transformer.drop)
        component["sparse_output"] = transformer.sparse_output
        # Categories (truncate if too long)
        if hasattr(transformer, "categories_"):
            n_categories = [len(cats) for cats in transformer.categories_]
            component["n_categories"] = n_categories
            # Only store first few categories per feature (full list goes to JSON)
            component["categories"] = [list(cats)[:50] for cats in transformer.categories_]

    elif class_name == "SimpleImputer":
        component["strategy"] = transformer.strategy
        if hasattr(transformer, "statistics_"):
            component["learned_stats"] = transformer.statistics_.tolist()

    elif class_name == "StandardScaler":
        if hasattr(transformer, "mean_"):
            component["mean"] = transformer.mean_.tolist()
        if hasattr(transformer, "scale_"):
            component["scale"] = transformer.scale_.tolist()

    elif class_name == "MinMaxScaler":
        if hasattr(transformer, "min_"):
            component["min"] = transformer.min_.tolist()
        if hasattr(transformer, "scale_"):
            component["scale"] = transformer.scale_.tolist()

    elif class_name == "RobustScaler":
        if hasattr(transformer, "center_"):
            component["center"] = transformer.center_.tolist()
        if hasattr(transformer, "scale_"):
            component["scale"] = transformer.scale_.tolist()

    # Recursively handle nested transformers
    if isinstance(transformer, Pipeline):
        component["steps"] = [_extract_component(step_name, step) for step_name, step in transformer.steps]

    return component


def compute_unknown_rates(artifact: Any, X: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compute unknown category rates for eval data (pre-transform).

    Args:
        artifact: Fitted preprocessing artifact
        X: Raw evaluation DataFrame

    Returns:
        Dictionary mapping feature -> {rate, count, top_unknowns}

    """
    unknown_rates: dict[str, dict[str, Any]] = {}

    # Find all OneHotEncoders
    encoders = _find_encoders(artifact)

    for encoder_info in encoders:
        encoder = encoder_info["encoder"]
        columns = encoder_info["columns"]

        if not hasattr(encoder, "categories_"):
            continue

        for i, col in enumerate(columns):
            if col not in X.columns:
                continue

            training_cats = set(encoder.categories_[i])
            eval_cats = set(X[col].dropna().unique())
            unknown_cats = eval_cats - training_cats

            if unknown_cats:
                n_unknown = X[col].isin(unknown_cats).sum()
                n_total = len(X)
                rate = n_unknown / n_total if n_total > 0 else 0.0

                unknown_rates[col] = {
                    "rate": rate,
                    "count": n_unknown,
                    "top_unknowns": sorted(unknown_cats)[:10],
                }

    return unknown_rates


def _find_encoders(artifact: Any) -> list[dict[str, Any]]:
    """Find all OneHotEncoders in artifact with their columns."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    encoders = []

    if isinstance(artifact, OneHotEncoder):
        encoders.append({"encoder": artifact, "columns": []})

    elif isinstance(artifact, Pipeline):
        for _, step in artifact.steps:
            encoders.extend(_find_encoders(step))

    elif isinstance(artifact, ColumnTransformer):
        for _, transformer, columns in artifact.transformers:
            if isinstance(transformer, OneHotEncoder):
                encoders.append({"encoder": transformer, "columns": columns})
            elif isinstance(transformer, Pipeline):
                # Find encoder in nested pipeline
                for _, step in transformer.steps:
                    if isinstance(step, OneHotEncoder):
                        encoders.append({"encoder": step, "columns": columns})

    return encoders
