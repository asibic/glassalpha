"""Test that pipeline enforces wrapper-only training with no direct estimator calls."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from glassalpha.config import AuditConfig
from glassalpha.pipeline import audit
from glassalpha.pipeline.audit import AuditPipeline

from ._utils.source_loader import get_module_source


def test_pipeline_uses_train_from_config_only():
    """Test that pipeline only uses train_from_config, never direct estimator training."""
    # Create minimal config for XGBoost
    config_dict = {
        "audit_profile": "test_profile",
        "reproducibility": {"random_seed": 42},
        "data": {
            "path": "dummy.csv",
            "target_column": "target",
            "protected_attributes": ["age"],
        },
        "model": {
            "type": "xgboost",
            "params": {
                "objective": "binary:logistic",
                "n_estimators": 10,
                "max_depth": 3,
            },
        },
        "explainers": {
            "strategy": "first_compatible",
            "priority": ["treeshap"],
        },
        "metrics": {
            "performance": ["accuracy"],
            "fairness": ["demographic_parity"],
        },
        "report": {
            "template": "standard_audit.html",
        },
    }

    config = AuditConfig(**config_dict)

    # Create mock data
    mock_data = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "age": np.random.randint(18, 80, 100),
            "target": np.random.randint(0, 2, 100),
        },
    )

    # Mock the data loader to return our test data
    with patch("glassalpha.data.tabular.TabularDataLoader.load") as mock_load:
        mock_load.return_value = mock_data

        # Mock _load_data to avoid file path validation
        with patch("glassalpha.pipeline.audit.AuditPipeline._load_data") as mock_load_data:
            mock_schema = MagicMock()
            mock_schema.features = ["feature1", "feature2"]
            mock_schema.target = "target"
            mock_load_data.return_value = (mock_data, mock_schema)

            # Mock train_from_config to track if it's called
            with patch("glassalpha.pipeline.train.train_from_config") as mock_train:
                mock_model = MagicMock()
                mock_model.get_capabilities.return_value = {"supports_shap": True}
                mock_model.predict.return_value = np.random.randint(0, 2, 100)
                mock_model.predict_proba.return_value = np.random.rand(100, 2)
                mock_train.return_value = mock_model

                # Mock other pipeline components
                with patch("glassalpha.pipeline.audit.AuditPipeline._select_explainer") as mock_explainer:
                    mock_explainer.return_value = MagicMock()

                    with patch("glassalpha.pipeline.audit.AuditPipeline._compute_metrics") as mock_metrics:
                        mock_metrics.return_value = None

                        # Create and run pipeline
                        pipeline = AuditPipeline(config)

                        # This should call train_from_config, not direct estimator training
                        try:
                            pipeline.run()
                        except Exception:
                            # We expect some failures due to mocking, but train_from_config should be called
                            pass

                        # Verify train_from_config was called
                        mock_train.assert_called_once()

                        # Verify the call signature
                        call_args = mock_train.call_args
                        assert call_args[0][0] == config  # First arg should be config
                        assert isinstance(call_args[0][1], pd.DataFrame)  # Second arg should be features
                        assert isinstance(call_args[0][2], (pd.Series, np.ndarray))  # Third arg should be target


def test_no_direct_estimator_imports_in_pipeline():
    """Test that pipeline modules don't import estimators directly."""
    # Get the source code for the audit module
    audit_content = get_module_source(audit, pkg_fallback="glassalpha.pipeline", filename="audit.py")

    # Check for forbidden direct estimator imports/usage
    forbidden_patterns = [
        "from xgboost import XGBClassifier",
        "from lightgbm import LGBMClassifier",
        "from sklearn.ensemble import RandomForestClassifier",
        "xgb.train(",
        "xgb.DMatrix(",
        "lgb.train(",
        "lgb.Dataset(",
    ]

    for pattern in forbidden_patterns:
        assert pattern not in audit_content, f"Pipeline contains forbidden direct estimator usage: {pattern}"


def test_wrapper_training_enforced():
    """Test that all model training goes through wrapper.fit() method."""
    # This test ensures that if someone tries to add direct estimator training,
    # it will be caught by checking that only wrapper methods are used

    config_dict = {
        "audit_profile": "test_profile",
        "reproducibility": {"random_seed": 42},
        "data": {
            "path": "dummy.csv",
            "target_column": "target",
            "protected_attributes": ["age"],
        },
        "model": {
            "type": "xgboost",
            "params": {"objective": "binary:logistic", "n_estimators": 10},
        },
        "explainers": {"strategy": "first_compatible", "priority": ["treeshap"]},
        "metrics": {"performance": ["accuracy"], "fairness": ["demographic_parity"]},
        "report": {"template": "standard_audit.html"},
    }

    config = AuditConfig(**config_dict)

    # Mock data
    mock_data = pd.DataFrame(
        {
            "feature1": np.random.randn(50),
            "age": np.random.randint(18, 80, 50),
            "target": np.random.randint(0, 2, 50),
        },
    )

    with patch("glassalpha.data.tabular.TabularDataLoader.load") as mock_load:
        mock_load.return_value = mock_data

        # Mock _load_data to avoid file path validation
        with patch("glassalpha.pipeline.audit.AuditPipeline._load_data") as mock_load_data:
            mock_schema = MagicMock()
            mock_schema.features = ["feature1"]
            mock_schema.target = "target"
            mock_load_data.return_value = (mock_data, mock_schema)

            # Patch the wrapper's fit method to track calls
            with patch("glassalpha.models.tabular.xgboost.XGBoostWrapper.fit") as mock_wrapper_fit:
                mock_model = MagicMock()
                mock_model.get_capabilities.return_value = {"supports_shap": True}
                mock_wrapper_fit.return_value = mock_model

                # Mock train_from_config to use the wrapper
                with patch("glassalpha.pipeline.train.train_from_config") as mock_train:
                    mock_train.return_value = mock_model

                    # Mock other components
                    with patch("glassalpha.pipeline.audit.AuditPipeline._select_explainer"):
                        with patch("glassalpha.pipeline.audit.AuditPipeline._compute_metrics"):
                            pipeline = AuditPipeline(config)

                            try:
                                pipeline.run()
                            except Exception:
                                pass  # Ignore failures due to mocking

                            # Verify that train_from_config was called (which should use wrapper)
                            mock_train.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
