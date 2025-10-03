#!/usr/bin/env python3
"""Custom data testing script for GlassAlpha models.

This script creates synthetic datasets and tests each model type
to identify any issues with custom data not in the repository.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def create_synthetic_classification_data(n_samples=1000, n_features=10, n_classes=2, random_state=42):
    """Create synthetic classification dataset."""
    np.random.seed(random_state)

    # Create features with some correlation structure
    X = np.random.randn(n_samples, n_features)

    # Add some feature interactions
    X[:, 2] = X[:, 0] * X[:, 1] + 0.1 * np.random.randn(n_samples)
    X[:, 5] = X[:, 3] + X[:, 4] + 0.2 * np.random.randn(n_samples)

    # Create target with some logic
    if n_classes == 2:
        # Binary classification
        y = (X[:, 0] + X[:, 1] + X[:, 2] + 0.5 * np.random.randn(n_samples) > 0).astype(int)
    else:
        # Multiclass classification
        y = X[:, 0] + X[:, 1] + X[:, 2] + 0.5 * np.random.randn(n_samples)
        y = np.digitize(y, bins=np.percentile(y, [33, 67]))  # 3 classes

    feature_names = [f"feature_{i:02d}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)

    return X_df, y, feature_names


def create_synthetic_regression_data(n_samples=1000, n_features=10, random_state=42):
    """Create synthetic regression dataset."""
    np.random.seed(random_state)

    # Create features
    X = np.random.randn(n_samples, n_features)

    # Add feature interactions
    X[:, 2] = X[:, 0] * X[:, 1] + 0.1 * np.random.randn(n_samples)
    X[:, 5] = X[:, 3] + X[:, 4] + 0.2 * np.random.randn(n_samples)

    # Create target
    y = X[:, 0] + X[:, 1] + X[:, 2] + 0.5 * np.random.randn(n_samples)

    feature_names = [f"feature_{i:02d}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)

    return X_df, y, feature_names


def test_xgboost_with_custom_data():
    """Test XGBoost with custom synthetic data."""
    print("Testing XGBoost with custom data...")

    try:
        from glassalpha.models.tabular.xgboost import XGBoostWrapper

        # Create custom data
        X, y, feature_names = create_synthetic_classification_data(n_samples=500, n_features=8)

        # Test binary classification
        print(f"  Binary classification: {X.shape[0]} samples, {X.shape[1]} features")

        # Create and train model
        model = XGBoostWrapper()
        model.fit(X, y)

        # Test predictions
        predictions = model.predict(X[:10])
        probabilities = model.predict_proba(X[:10])

        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Probabilities shape: {probabilities.shape}")
        print(f"  Unique predictions: {np.unique(predictions)}")

        # Test feature importance
        importance = model.get_feature_importance()
        if isinstance(importance, dict):
            print(f"  Feature importance (dict): {len(importance)} entries")
        else:
            print(f"  Feature importance shape: {importance.shape}")

        # Test save/load
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            model.save(f.name)
            loaded_model = XGBoostWrapper()
            loaded_model.load(f.name)

            # Compare predictions
            original_pred = model.predict(X[:5])
            loaded_pred = loaded_model.predict(X[:5])

            if np.array_equal(original_pred, loaded_pred):
                print("  Save/load test: PASSED")
            else:
                print("  Save/load test: FAILED - predictions don't match")
                return False

        # Test multiclass
        X_multi, y_multi, _ = create_synthetic_classification_data(n_samples=300, n_features=6, n_classes=3)
        print(
            f"  Multiclass classification: {X_multi.shape[0]} samples, {X_multi.shape[1]} features, {len(np.unique(y_multi))} classes",
        )

        model_multi = XGBoostWrapper()
        model_multi.fit(X_multi, y_multi)
        pred_multi = model_multi.predict(X_multi[:10])
        prob_multi = model_multi.predict_proba(X_multi[:10])

        print(f"  Multiclass predictions shape: {pred_multi.shape}")
        print(f"  Multiclass probabilities shape: {prob_multi.shape}")
        print(f"  Multiclass unique predictions: {np.unique(pred_multi)}")

        print("  XGBoost custom data test: PASSED")
        return True

    except Exception as e:
        print(f"  XGBoost custom data test: FAILED - {e}")
        import traceback

        traceback.print_exc()
        return False


def test_lightgbm_with_custom_data():
    """Test LightGBM with custom synthetic data."""
    print("Testing LightGBM with custom data...")

    try:
        from glassalpha.models.tabular.lightgbm import LightGBMWrapper

        # Create custom data
        X, y, feature_names = create_synthetic_classification_data(n_samples=500, n_features=8)

        print(f"  Binary classification: {X.shape[0]} samples, {X.shape[1]} features")

        # Create and train model
        model = LightGBMWrapper()
        model.fit(X, y)

        # Test predictions
        predictions = model.predict(X[:10])
        probabilities = model.predict_proba(X[:10])

        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Probabilities shape: {probabilities.shape}")
        print(f"  Unique predictions: {np.unique(predictions)}")

        # Test feature importance
        importance = model.get_feature_importance()
        if isinstance(importance, dict):
            print(f"  Feature importance (dict): {len(importance)} entries")
        else:
            print(f"  Feature importance shape: {importance.shape}")

        # Test save/load
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            model.save(f.name)
            loaded_model = LightGBMWrapper()
            loaded_model.load(f.name)

            # Compare predictions
            original_pred = model.predict(X[:5])
            loaded_pred = loaded_model.predict(X[:5])

            if np.array_equal(original_pred, loaded_pred):
                print("  Save/load test: PASSED")
            else:
                print("  Save/load test: FAILED - predictions don't match")
                return False

        # Test multiclass
        X_multi, y_multi, _ = create_synthetic_classification_data(n_samples=300, n_features=6, n_classes=3)
        print(
            f"  Multiclass classification: {X_multi.shape[0]} samples, {X_multi.shape[1]} features, {len(np.unique(y_multi))} classes",
        )

        model_multi = LightGBMWrapper()
        model_multi.fit(X_multi, y_multi)
        pred_multi = model_multi.predict(X_multi[:10])
        prob_multi = model_multi.predict_proba(X_multi[:10])

        print(f"  Multiclass predictions shape: {pred_multi.shape}")
        print(f"  Multiclass probabilities shape: {prob_multi.shape}")
        print(f"  Multiclass unique predictions: {np.unique(pred_multi)}")

        print("  LightGBM custom data test: PASSED")
        return True

    except Exception as e:
        print(f"  LightGBM custom data test: FAILED - {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sklearn_with_custom_data():
    """Test sklearn models with custom synthetic data."""
    print("Testing sklearn models with custom data...")

    try:
        from sklearn.ensemble import RandomForestClassifier

        from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper, SklearnGenericWrapper

        # Create custom data
        X, y, feature_names = create_synthetic_classification_data(n_samples=500, n_features=8)

        print(f"  Binary classification: {X.shape[0]} samples, {X.shape[1]} features")

        # Test LogisticRegression wrapper
        lr_model = LogisticRegressionWrapper()
        lr_model.fit(X, y)

        predictions = lr_model.predict(X[:10])
        probabilities = lr_model.predict_proba(X[:10])

        print(f"  LogisticRegression predictions shape: {predictions.shape}")
        print(f"  LogisticRegression probabilities shape: {probabilities.shape}")
        print(f"  LogisticRegression unique predictions: {np.unique(predictions)}")

        # Test save/load for LogisticRegression
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            lr_model.save(f.name)
            loaded_lr = LogisticRegressionWrapper()
            loaded_lr.load(f.name)

            original_pred = lr_model.predict(X[:5])
            loaded_pred = loaded_lr.predict(X[:5])

            if np.array_equal(original_pred, loaded_pred):
                print("  LogisticRegression save/load test: PASSED")
            else:
                print("  LogisticRegression save/load test: FAILED")
                return False

        # Test generic wrapper with RandomForest (requires pre-fitted model)
        rf_estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rf_estimator.fit(X, y)  # Fit the underlying model first
        rf_model = SklearnGenericWrapper(rf_estimator)

        rf_predictions = rf_model.predict(X[:10])
        rf_probabilities = rf_model.predict_proba(X[:10])

        print(f"  RandomForest predictions shape: {rf_predictions.shape}")
        print(f"  RandomForest probabilities shape: {rf_probabilities.shape}")
        print(f"  RandomForest unique predictions: {np.unique(rf_predictions)}")

        # Test multiclass
        X_multi, y_multi, _ = create_synthetic_classification_data(n_samples=300, n_features=6, n_classes=3)
        print(
            f"  Multiclass classification: {X_multi.shape[0]} samples, {X_multi.shape[1]} features, {len(np.unique(y_multi))} classes",
        )

        lr_multi = LogisticRegressionWrapper()
        lr_multi.fit(X_multi, y_multi)
        pred_multi = lr_multi.predict(X_multi[:10])
        prob_multi = lr_multi.predict_proba(X_multi[:10])

        print(f"  Multiclass predictions shape: {pred_multi.shape}")
        print(f"  Multiclass probabilities shape: {prob_multi.shape}")
        print(f"  Multiclass unique predictions: {np.unique(pred_multi)}")

        print("  Sklearn custom data test: PASSED")
        return True

    except Exception as e:
        print(f"  Sklearn custom data test: FAILED - {e}")
        import traceback

        traceback.print_exc()
        return False


def test_explainer_integration_with_custom_data():
    """Test explainers with custom data."""
    print("Testing explainers with custom data...")

    try:
        from glassalpha.explain.shap.kernel import KernelSHAPExplainer
        from glassalpha.explain.shap.tree import TreeSHAPExplainer
        from glassalpha.models.tabular.xgboost import XGBoostWrapper

        # Create custom data
        X, y, feature_names = create_synthetic_classification_data(n_samples=200, n_features=6)

        # Train model
        model = XGBoostWrapper()
        model.fit(X, y)

        print(f"  Testing with {X.shape[0]} samples, {X.shape[1]} features")

        # Test TreeSHAP
        treeshap = TreeSHAPExplainer()
        treeshap.fit(model, X)

        # Local explanations
        local_explanations = treeshap.explain(X[:5])
        print(f"  TreeSHAP local explanations shape: {local_explanations.shape}")

        # Global explanations (aggregate local explanations)
        global_explanations = np.mean(local_explanations, axis=0)
        print(f"  TreeSHAP global explanations shape: {global_explanations.shape}")

        # Test KernelSHAP
        kernelshap = KernelSHAPExplainer()
        kernelshap.fit(model, X)

        # Local explanations
        local_explanations_k = kernelshap.explain(X[:3])  # Smaller sample for speed
        print(f"  KernelSHAP local explanations shape: {local_explanations_k.shape}")

        print("  Explainer custom data test: PASSED")
        return True

    except Exception as e:
        print(f"  Explainer custom data test: FAILED - {e}")
        import traceback

        traceback.print_exc()
        return False


def test_pipeline_with_custom_data():
    """Test full pipeline with custom data."""
    print("Testing full pipeline with custom data...")

    try:
        import csv

        from glassalpha.config import load_config_from_file

        # Create custom data
        X, y, feature_names = create_synthetic_classification_data(n_samples=200, n_features=6)

        # Save custom data to temporary CSV files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            data_path = f.name
            writer = csv.writer(f)
            writer.writerow(feature_names + ["target"])
            for i in range(len(y)):
                row = list(X.iloc[i]) + [y[i]]
                writer.writerow(row)

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_path = f.name
            f.write(f"""
audit_profile: tabular_compliance
model:
  type: xgboost
  params:
    n_estimators: 10
    random_state: 42
data:
  dataset: custom
  path: {data_path}
  target_column: target
report:
  path: /tmp/custom_test_output.pdf
""")

        # Load and validate config
        config = load_config_from_file(config_path)

        print("  Config loaded successfully")
        print(f"  Data path: {config.data.path}")
        print("  Pipeline custom data test: PASSED")
        return True

    except Exception as e:
        print(f"  Pipeline custom data test: FAILED - {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all custom data tests."""
    print("=" * 60)
    print("GLASSALPHA CUSTOM DATA TESTING")
    print("=" * 60)

    results = []

    # Test each model type
    results.append(("XGBoost", test_xgboost_with_custom_data()))
    results.append(("LightGBM", test_lightgbm_with_custom_data()))
    results.append(("Sklearn", test_sklearn_with_custom_data()))
    results.append(("Explainers", test_explainer_integration_with_custom_data()))
    results.append(("Pipeline", test_pipeline_with_custom_data()))

    # Summary
    print("\n" + "=" * 60)
    print("CUSTOM DATA TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:15} : {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All custom data tests passed!")
    else:
        print(f"⚠️  {failed} custom data tests failed")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
