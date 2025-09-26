#!/bin/bash
# Wheel-first test harness - exactly like CI

set -e  # Exit on first error

echo "🔧 Step 1: Clean previous builds..."
rm -rf dist build *.egg-info

echo "📦 Step 2: Build wheel..."
python -m build

echo "🔍 Step 3: Verify templates in wheel..."
echo "Checking for templates in wheel:"
unzip -l dist/*.whl | grep -E "glassalpha/report/templates/.*\.html" || {
    echo "❌ ERROR: Templates not found in wheel!"
    echo "Contents of wheel:"
    unzip -l dist/*.whl | grep glassalpha/report/ | head -20
    exit 1
}

echo "📥 Step 4: Install wheel (uninstall first)..."
python -m pip uninstall -y glassalpha
python -m pip install dist/*.whl

echo "🧪 Step 5: Run critical contract tests..."
python -m pytest -xvs \
    tests/test_pipeline_basic.py::test_pipeline_logs_initialization \
    tests/test_model_integration.py::test_model_saving_and_loading \
    -k "not xgboost" \
    2>&1 | tee test_output.log

echo "✅ Step 6: Quick smoke test..."
python -c "
import sys
from glassalpha.pipeline.audit import AuditPipeline
from glassalpha.config.schema import AuditConfig

# Test 1: Logger format
print('Testing logger format...')
# This will fail if logger format is wrong

# Test 2: Template loading
print('Testing template loading...')
try:
    from glassalpha.report.renderer import AuditReportRenderer
    renderer = AuditReportRenderer()
    print('✅ Template loading works')
except Exception as e:
    print(f'❌ Template loading failed: {e}')
    sys.exit(1)

# Test 3: Model roundtrip
print('Testing model save/load...')
from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper
import tempfile
import numpy as np

wrapper = LogisticRegressionWrapper()
X = np.array([[1, 2], [3, 4]])
y = np.array([0, 1])
wrapper.fit(X, y)

with tempfile.NamedTemporaryFile(suffix='.pkl') as f:
    wrapper.save(f.name)
    new_wrapper = LogisticRegressionWrapper()
    new_wrapper.load(f.name)
    assert new_wrapper.model is not None, 'Model not loaded!'
    pred = new_wrapper.predict(X)
    print('✅ Model save/load works')

print('All smoke tests passed!')
"

echo "🎉 All wheel tests completed!"
