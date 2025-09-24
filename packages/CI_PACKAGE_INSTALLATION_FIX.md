# CI Package Installation Issue Fix

## 🚨 Problem Identified

CI failing with:
```
ModuleNotFoundError: No module named 'glassalpha.data'
```

Affects:
- `tests/test_data_loading.py`
- `tests/test_end_to_end.py`
- `tests/test_pipeline_basic.py`

## 🔍 Analysis

**Local Setup (Working):**
- Editable install: `pip install -e .`
- Location: `/Users/gabe/Sites/glassalpha/packages/venv/lib/python3.13/site-packages`
- Package structure: All modules properly accessible

**CI Setup (Failing):**
- Package not installing properly
- Missing editable install step?
- Path/PYTHONPATH issues?

## 🛠️ Possible Solutions

### Option 1: Add Explicit Package Installation
CI likely needs:
```bash
cd packages
pip install -e .  # Editable install
```

### Option 2: Add PYTHONPATH
If install not working, fallback:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/packages/src"
```

### Option 3: Install Step Missing
CI workflow may be missing the package installation step entirely.

## 🎯 Expected Fix

After adding proper package installation to CI:
- All 214 tests should be collectible
- Only sklearn-dependent tests should skip gracefully
- Coverage should jump to ~200+ tests passing

## 🚨 CI Workflow Needs

The CI workflow should include:
```yaml
- name: Install package
  run: |
    cd packages
    pip install -e .
```

Before running pytest.

## Status

- ✅ Conditional imports fixed (test collection ready)
- ❌ Package installation issue (CI configuration)
- 🎯 Ready for CI workflow fix
