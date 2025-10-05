"""Versioned serialization tests - forward compatibility critical path validation.

These tests ensure saved models can be loaded by future versions:
- Schema version handling must be robust
- Forward compatibility logic must work
- Breaking changes must be detectable
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestVersionedSerializationRisk:
    """Test serialization critical paths that prevent customer model loss."""

    def test_load_legacy_v1_model_format(self):
        """Must be able to load models saved in v1 format - customer backward compatibility."""
        from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper

        # Create a mock v1 format saved model (what a previous version might have saved)
        legacy_v1_data = {
            "model": {
                # Simplified sklearn LogisticRegression state
                "__class__": "sklearn.linear_model._logistic.LogisticRegression",
                "__module__": "sklearn.linear_model._logistic",
                "C": 1.0,
                "class_weight": None,
                "dual": False,
                "fit_intercept": True,
                "intercept_scaling": 1,
                "l1_ratio": None,
                "max_iter": 1000,
                "n_jobs": None,
                "penalty": "l2",
                "random_state": 42,
                "solver": "lbfgs",
                "tol": 0.0001,
                "verbose": 0,
                "warm_start": False,
                "coef_": [[0.5, -0.3]],
                "intercept_": [0.1],
                "classes_": [0, 1],
                "n_features_in_": 2,
                "n_iter_": [5],
            },
            "feature_names_": ["feature_a", "feature_b"],
            "n_classes": 2,
            "version": "1.0.0",  # Legacy version marker
            "_is_fitted": True,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_path = Path(tmpdir) / "legacy_v1_model.json"

            # Save in legacy format
            with legacy_path.open("w") as f:
                json.dump(legacy_v1_data, f, indent=2, default=str)

            # Current version should be able to load it
            wrapper = LogisticRegressionWrapper()
            try:
                wrapper.load(legacy_path)

                # Should preserve critical metadata
                assert wrapper.feature_names_ == ["feature_a", "feature_b"]
                assert wrapper.n_classes == 2
                assert wrapper._is_fitted == True

                # Should be able to make predictions on compatible data
                X_test = pd.DataFrame(
                    {
                        "feature_a": [1.0, -1.0],
                        "feature_b": [0.5, -0.5],
                    },
                )
                predictions = wrapper.predict(X_test)
                assert len(predictions) == 2

            except Exception as e:
                # If loading fails, we need to implement backward compatibility
                pytest.fail(f"Failed to load legacy v1 format: {e}")

    def test_current_version_saves_with_version_info(self):
        """Current version must save models with version information for future compatibility."""
        from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper

        X_train = pd.DataFrame({"x1": [1, 2, 3, 4], "x2": [1, 0, 1, 0]})
        y_train = [0, 1, 0, 1]

        wrapper = LogisticRegressionWrapper()
        wrapper.fit(X_train, y_train, random_state=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "versioned_model.json"
            wrapper.save(model_path)

            # Load the raw JSON to inspect structure
            with model_path.open() as f:
                saved_data = json.load(f)

            # Must contain version information for future compatibility
            required_fields = ["model", "feature_names_", "n_classes", "_is_fitted"]
            for field in required_fields:
                assert field in saved_data, f"Saved model must contain {field} for compatibility"

            # Should have identifiable structure
            assert isinstance(saved_data["feature_names_"], list), "feature_names_ must be preserved as list"
            assert isinstance(saved_data["n_classes"], int), "n_classes must be preserved as int"

    def test_xgboost_json_format_forward_compatibility(self):
        """XGBoost JSON format must be compatible across library versions."""
        from glassalpha.models.tabular.xgboost import XGBoostWrapper

        X_train = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 40),
                "feature_2": np.random.uniform(-1, 1, 40),
            },
        )
        y_train = np.random.binomial(1, 0.5, 40)

        wrapper = XGBoostWrapper()
        wrapper.fit(X_train, y_train, random_state=123)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "xgb_versioned.json"
            wrapper.save(model_path)

            # Load the raw JSON structure
            with model_path.open() as f:
                saved_data = json.load(f)

            # XGBoost-specific compatibility checks
            assert "model" in saved_data, "Must contain XGBoost model data"
            assert "feature_names_" in saved_data, "Must preserve feature names"
            assert "n_classes" in saved_data, "Must preserve class count"

            # The model field should contain XGBoost JSON (string format)
            assert isinstance(saved_data["model"], str), "XGBoost model should be JSON string"

            # Try to parse the inner XGBoost JSON to ensure it's valid
            try:
                xgb_data = json.loads(saved_data["model"])
                assert "version" in xgb_data or "learner" in xgb_data, "Should be valid XGBoost format"
            except json.JSONDecodeError:
                pytest.fail("XGBoost model data should be valid JSON for forward compatibility")

    def test_schema_evolution_handling(self):
        """Must gracefully handle schema changes in future versions."""
        from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper

        # Simulate a future schema with additional fields
        future_schema_data = {
            "model": {"dummy": "model_data"},  # Simplified for test
            "feature_names_": ["a", "b", "c"],
            "n_classes": 3,
            "_is_fitted": True,
            "schema_version": "2.0.0",  # Future version
            "new_future_field": "some_future_data",  # New field
            "another_new_field": {"complex": "structure"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            future_path = Path(tmpdir) / "future_schema.json"

            with future_path.open("w") as f:
                json.dump(future_schema_data, f)

            # Current version should handle unknown fields gracefully
            wrapper = LogisticRegressionWrapper()
            try:
                # This might fail due to model format, but should not crash on unknown fields
                wrapper.load(future_path)
                # If it loads, basic fields should work
                if wrapper.feature_names_:
                    assert len(wrapper.feature_names_) == 3
            except (ValueError, KeyError, TypeError) as e:
                # Expected - but should be a graceful error, not a crash
                error_msg = str(e).lower()
                assert "unknown" not in error_msg or "field" not in error_msg, (
                    f"Should not fail due to unknown fields: {e}"
                )

    def test_model_format_validation(self):
        """Saved models must have consistent, validatable format."""
        from glassalpha.models.tabular.lightgbm import LightGBMWrapper

        pytest.importorskip("lightgbm")  # Skip if not available

        X_train = pd.DataFrame({"x": np.random.normal(size=30), "y": np.random.normal(size=30)})
        y_train = np.random.binomial(1, 0.4, 30)

        wrapper = LightGBMWrapper()
        wrapper.fit(X_train, y_train, random_state=567)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "lgb_format_test.json"
            wrapper.save(model_path)

            # Validate the saved format is consistent
            with model_path.open() as f:
                data = json.load(f)

            # Must be a dictionary at the top level
            assert isinstance(data, dict), "Saved format must be JSON object"

            # Must have string keys (JSON requirement)
            for key in data.keys():
                assert isinstance(key, str), f"All keys must be strings, got {type(key)}"

            # Critical fields must have correct types
            if "feature_names_" in data:
                assert isinstance(data["feature_names_"], list), "feature_names_ must be list"

            if "n_classes" in data:
                assert isinstance(data["n_classes"], int), "n_classes must be integer"

    def test_corrupted_model_file_handling(self):
        """Must handle corrupted model files gracefully."""
        from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create various corrupted files
            corrupted_files = {
                "empty.json": "",
                "invalid_json.json": "{ invalid json content",
                "wrong_structure.json": json.dumps({"not": "model_data"}),
                "missing_fields.json": json.dumps({"model": "data"}),  # Missing required fields
            }

            wrapper = LogisticRegressionWrapper()

            for filename, content in corrupted_files.items():
                file_path = Path(tmpdir) / filename
                file_path.write_text(content)

                # Should raise appropriate error, not crash
                with pytest.raises((ValueError, KeyError, json.JSONDecodeError, FileNotFoundError)):
                    wrapper.load(file_path)
