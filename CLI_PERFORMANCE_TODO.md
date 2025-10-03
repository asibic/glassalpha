# CLI Performance Optimization Checklist

**Goal**: Reduce `glassalpha --help` from 635ms to <300ms (warm cache) **AND** fix explainer registry API bug

**Current Status**: 635ms → Target: <300ms | Explainer crash: BROKEN → FIXED

---

## ⚠️ CRITICAL: Phase 2.5 First - Fix Explainer API Bug (45-60 min)

**Why this is first**: The explainer registry has a hard API bug that crashes audits. Performance improvements don't matter if the tool doesn't work.

**The Bug**: `TreeSHAPExplainer.is_compatible` has wrong signature. Registry calls `cls.is_compatible(model=...)` but explainer expects different args.

### Step 2.5.1: Unify the Explainer Contract

- [ ] **2.5.1.1** Update `packages/src/glassalpha/explain/base.py`

  - [ ] Open the base explainer class
  - [ ] Add or update the interface specification:

    ```python
    class BaseExplainer:
        @classmethod
        def is_compatible(cls, *, model=None, model_type=None, config=None) -> bool:
            """Check if this explainer is compatible with the given model.

            Args:
                model: The model instance (optional)
                model_type: String model type identifier (optional)
                config: Configuration dict (optional)

            Returns:
                True if compatible, False otherwise

            Note:
                All arguments are keyword-only to prevent signature drift.
            """
            raise NotImplementedError
    ```

  - [ ] Document that all args are keyword-only and optional

### Step 2.5.2: Update All Explainers to Match

- [ ] **2.5.2.1** Fix TreeSHAPExplainer

  - [ ] Open `packages/src/glassalpha/explain/shap/tree.py`
  - [ ] Find `is_compatible` method
  - [ ] Change signature to: `@classmethod def is_compatible(cls, *, model=None, model_type=None, config=None) -> bool:`
  - [ ] Update implementation to check for tree models using the new signature
  - [ ] Example logic:
    ```python
    if model_type:
        return model_type in ["xgboost", "lightgbm", "random_forest", "decision_tree"]
    if model is not None:
        # Check model instance type
        return hasattr(model, "get_booster") or hasattr(model, "tree_")
    return False
    ```

- [ ] **2.5.2.2** Fix KernelSHAPExplainer

  - [ ] Open `packages/src/glassalpha/explain/shap/kernel.py`
  - [ ] Update `is_compatible` to match the signature
  - [ ] Should be compatible with most models (return True as fallback)

- [ ] **2.5.2.3** Fix NoOpExplainer

  - [ ] Open `packages/src/glassalpha/explain/noop.py`
  - [ ] Update `is_compatible` signature
  - [ ] Should return True (always compatible)

- [ ] **2.5.2.4** Fix CoefficientsExplainer
  - [ ] Open `packages/src/glassalpha/explain/coefficients.py`
  - [ ] Update `is_compatible` signature
  - [ ] Check for linear models (logistic_regression, linear_model)

### Step 2.5.3: Make Registry Tolerant but Strict

- [ ] **2.5.3.1** Update explainer registry selection logic
  - [ ] Open `packages/src/glassalpha/core/registry.py` or wherever selection happens
  - [ ] Find where `is_compatible` is called
  - [ ] Wrap in try/except:
    ```python
    try:
        # Try with keywords first (new API)
        if explainer_cls.is_compatible(model=model, model_type=model_type, config=config):
            return explainer_cls
    except TypeError as e:
        # Log warning about legacy signature
        logger.warning(f"{explainer_cls.__name__}.is_compatible has wrong signature: {e}")
        # Try legacy fallback if needed, or treat as incompatible
        try:
            if explainer_cls.is_compatible(model):
                logger.warning(f"Using legacy is_compatible() for {explainer_cls.__name__}")
                return explainer_cls
        except Exception:
            pass  # Treat as incompatible
    except Exception as e:
        logger.error(f"Error checking {explainer_cls.__name__} compatibility: {e}")
        # Don't crash - just skip this explainer
    ```

### Step 2.5.4: Add Registry Contract Tests

- [ ] **2.5.4.1** Create `packages/tests/test_explainer_registry_contract.py`

  - [ ] Add test that iterates all registered explainers:

    ```python
    import pytest
    from glassalpha.core.registry import ExplainerRegistry

    def test_all_explainers_have_correct_is_compatible_signature():
        """Ensure all registered explainers implement is_compatible correctly."""
        ExplainerRegistry.discover()

        for name, explainer_cls in ExplainerRegistry._registry.items():
            # Should not raise TypeError with keyword args
            try:
                result = explainer_cls.is_compatible(
                    model=None,
                    model_type="test",
                    config={}
                )
                assert isinstance(result, bool), \
                    f"{name}.is_compatible should return bool, got {type(result)}"
            except TypeError as e:
                pytest.fail(f"{name}.is_compatible has wrong signature: {e}")
    ```

- [ ] **2.5.4.2** Run the contract test
  - [ ] `cd packages && pytest tests/test_explainer_registry_contract.py -v`
  - [ ] Fix any explainers that fail

### Step 2.5.5: Add End-to-End Smoke Test

- [ ] **2.5.5.1** Create `packages/tests/test_audit_smoke.py`

  - [ ] Add smoke test for german_credit audit:

    ```python
    import subprocess
    import sys
    from pathlib import Path
    import pytest

    def test_audit_german_credit_simple_works(tmp_path):
        """Smoke test: full audit should complete without crashing."""
        pdf = tmp_path / "audit.pdf"
        config_path = Path(__file__).parent.parent / "configs" / "german_credit_simple.yaml"

        result = subprocess.run(
            [sys.executable, "-m", "glassalpha", "audit",
             "--config", str(config_path),
             "--output", str(pdf)],
            capture_output=True,
            text=True
        )

        # Should succeed
        assert result.returncode == 0, \
            f"Audit failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"

        # PDF should exist and have reasonable size
        assert pdf.exists(), "PDF was not generated"
        assert pdf.stat().st_size > 10_000, "PDF seems too small"
    ```

- [ ] **2.5.5.2** Test the smoke test

  - [ ] `cd packages && pytest tests/test_audit_smoke.py -v -s`
  - [ ] Verify audit completes successfully

- [ ] **2.5.5.3** Validate explainer was selected correctly
  - [ ] Check logs/output to confirm correct explainer was used
  - [ ] Verify no TypeError or signature errors in output

---

## Phase 1: Quick Win - sklearn Import (30 min) ⚡

**Expected Impact**: 635ms → ~270ms (58% speedup)

- [ ] **1.1** Move sklearn import in `data/tabular.py`

  - [ ] Open `packages/src/glassalpha/data/tabular.py`
  - [ ] Find line 19: `from sklearn.model_selection import train_test_split`
  - [ ] Remove this line from module level
  - [ ] Find `split_data()` method (around line 350)
  - [ ] Add `from sklearn.model_selection import train_test_split` at start of method
  - [ ] Add comment: `# Lazy import - don't load sklearn during CLI --help`

- [ ] **1.2** Test the change

  - [ ] Run: `cd packages && source venv/bin/activate`
  - [ ] Run: `time glassalpha --help` (should be ~270ms)
  - [ ] Run: `time glassalpha --version` (should be <50ms)
  - [ ] Run: `pytest tests/test_data_tabular.py -v` (ensure tests pass)

- [ ] **1.3** Verify no regressions
  - [ ] Run full test suite: `pytest tests/ -v`
  - [ ] Check that audit still works: `glassalpha audit --config configs/quickstart.yaml --out test.pdf`
  - [ ] Delete test.pdf: `rm test.pdf test.manifest.json`

---

## Phase 2: Implement True Lazy Command Loading (1-2 hours) 🎯

**Expected Impact**: ~270ms → <200ms

**Key Insight**: Typer/Click eagerly import subcommands when rendering help unless you use a lazy group.

### Step 2.1: Implement LazyGroup for Click/Typer

- [ ] **2.1.1** Create lazy loading infrastructure

  - [ ] Open `packages/src/glassalpha/cli/main.py`
  - [ ] Add at top of file:

    ```python
    import importlib
    from typing import Any

    class LazyGroup(typer.Typer):
        """Typer group that lazily imports commands only when invoked."""

        def __init__(self, *args: Any, lazy_commands: dict[str, str] | None = None, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._lazy_commands = lazy_commands or {}

        def add_lazy_command(self, name: str, module_path: str, func_name: str) -> None:
            """Register a command that will be imported lazily."""
            self._lazy_commands[name] = (module_path, func_name)

            # Create a placeholder command for help text
            @self.command(name)
            def lazy_wrapper(*args: Any, **kwargs: Any) -> Any:
                """Lazy command wrapper."""
                module = importlib.import_module(module_path)
                func = getattr(module, func_name)
                return func(*args, **kwargs)
    ```

### Step 2.2: Convert Datasets Commands to Lazy Loading

- [ ] **2.2.1** Remove eager imports from cli/main.py

  - [ ] Find line 109: `from .datasets import list_datasets, dataset_info, show_cache_dir, fetch_dataset`
  - [ ] Delete this line

- [ ] **2.2.2** Replace with lazy command registration

  - [ ] Find the datasets_app command registrations (lines 117-120)
  - [ ] Replace with:

    ```python
    # Lazy dataset commands - don't import until invoked
    @datasets_app.command("list")
    def list_datasets_lazy():
        """List all available datasets in the registry."""
        from .datasets import list_datasets
        return list_datasets()

    @datasets_app.command("info")
    def dataset_info_lazy(dataset: str = typer.Argument(..., help="Dataset key to inspect")):
        """Show detailed information about a specific dataset."""
        from .datasets import dataset_info
        return dataset_info(dataset)

    @datasets_app.command("cache-dir")
    def show_cache_dir_lazy():
        """Show the directory where datasets are cached."""
        from .datasets import show_cache_dir
        return show_cache_dir()

    @datasets_app.command("fetch")
    def fetch_dataset_lazy(
        dataset: str = typer.Argument(..., help="Dataset key to fetch"),
        force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
        dest: Path = typer.Option(None, "--dest", help="Custom destination path"),
    ):
        """Fetch and cache a dataset from the registry."""
        from .datasets import fetch_dataset
        return fetch_dataset(dataset, force, dest)
    ```

- [ ] **2.2.3** Test dataset commands
  - [ ] Run: `time glassalpha --help` (should be improving)
  - [ ] Run: `glassalpha datasets list`
  - [ ] Run: `glassalpha datasets info german_credit`
  - [ ] Run: `glassalpha datasets cache-dir`

### Step 2.3: Fix datasets/**init**.py with **getattr** Pattern

- [ ] **2.3.1** Make datasets/**init**.py lazy but backward compatible

  - [ ] Open `packages/src/glassalpha/datasets/__init__.py`
  - [ ] Remove eager imports (lines 7-11)
  - [ ] Replace with lazy `__getattr__`:

    ```python
    """Dataset loaders for common benchmark datasets."""

    from .registry import REGISTRY, DatasetSpec

    __all__ = [
        "REGISTRY",
        "DatasetSpec",
        "GermanCreditDataset",
        "get_german_credit_schema",
        "load_german_credit",
    ]

    def __getattr__(name: str):
        """Lazy import dataset loaders to avoid heavy imports during CLI --help."""
        if name == "GermanCreditDataset":
            from .german_credit import GermanCreditDataset
            return GermanCreditDataset
        if name == "get_german_credit_schema":
            from .german_credit import get_german_credit_schema
            return get_german_credit_schema
        if name == "load_german_credit":
            from .german_credit import load_german_credit
            return load_german_credit
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Don't register datasets eagerly - let them register on first use
    # register_builtin_datasets()  # REMOVED
    ```

- [ ] **2.3.2** Update dataset registration to be lazy

  - [ ] Check where `register_builtin_datasets()` is needed
  - [ ] Call it only when datasets are actually accessed
  - [ ] Or call it inside dataset commands

- [ ] **2.3.3** Test backward compatibility
  - [ ] Test that `from glassalpha.datasets import load_german_credit` still works
  - [ ] Test that existing code doesn't break
  - [ ] Run dataset tests: `pytest tests/test_datasets* -v`

---

## Phase 3: Audit All Module-Level Imports (1 hour) 🔍

**Updated approach**: Don't break user-facing imports, use `__getattr__` pattern

- [ ] **3.1** Find all problematic imports

  - [ ] Run audit script:
    ```bash
    cd packages
    grep -rn "^from.*import" src/glassalpha/ | \
      grep -E "(pandas|numpy|matplotlib|weasyprint|xgboost|lightgbm|shap|sklearn)" | \
      grep -v "# lazy" | \
      grep -v "TYPE_CHECKING" | \
      grep -v "test" > /tmp/cli_imports.txt
    cat /tmp/cli_imports.txt
    ```

- [ ] **3.2** Review each import found

  - [ ] Check if import is in `__init__.py` → HIGH PRIORITY (use `__getattr__`)
  - [ ] Check if import is in `cli/` directory → HIGH PRIORITY (move to function)
  - [ ] Check if import is at module level in data loaders → MEDIUM PRIORITY
  - [ ] Create list of files to fix

- [ ] **3.3** Fix each problematic import

  - [ ] For `__init__.py` files: Use `__getattr__` pattern (preserve API)
  - [ ] For other files: Move import inside function/method
  - [ ] Add comment: `# Lazy import - avoid loading during CLI startup`
  - [ ] Test after each fix

- [ ] **3.4** Special attention to `__init__.py` files
  - [ ] `src/glassalpha/__init__.py` - check and fix
  - [ ] `src/glassalpha/data/__init__.py` - check and fix
  - [ ] `src/glassalpha/models/__init__.py` - check and fix
  - [ ] `src/glassalpha/explain/__init__.py` - check and fix

---

## Phase 5: Add Performance Guards (1 hour) 🧪

**Updated with specific metrics from feedback**

### Step 5.1: Create Comprehensive Performance Tests

- [ ] **5.1.1** Create `packages/tests/test_cli_performance.py`

  - [ ] Add help latency test:

    ```python
    import subprocess
    import sys
    import time
    import os

    def test_help_is_fast():
        """Ensure --help renders quickly."""
        # Warm up
        subprocess.run([sys.executable, "-m", "glassalpha", "--version"],
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Measure help
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, "-m", "glassalpha", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        dt = time.time() - t0

        # Start loose, tighten later
        threshold = 0.60 if os.getenv("CI") else 0.30
        assert dt < threshold, f"--help took {dt:.3f}s, should be <{threshold}s"
        assert "GlassAlpha" in result.stdout

    def test_version_is_instant():
        """Version should be nearly instant."""
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, "-m", "glassalpha", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        dt = time.time() - t0
        assert dt < 0.10, f"--version took {dt:.3f}s"
    ```

- [ ] **5.1.2** Add import-time profiling test

  - [ ] Add to same file:

    ```python
    def test_import_time_is_reasonable(tmp_path):
        """Check total import time using Python's -X importtime."""
        import_log = tmp_path / "importtime.txt"

        result = subprocess.run(
            [sys.executable, "-X", "importtime", "-c", "import glassalpha.cli.main"],
            stderr=subprocess.PIPE,
            text=True
        )

        import_log.write_text(result.stderr)

        # Parse the last line for total time
        lines = result.stderr.strip().split('\n')
        if lines:
            # Format: "import time: self | cumulative | <module>"
            # Last line has total time
            last = lines[-1]
            # Extract cumulative time (in microseconds usually)
            # This is approximate - adjust based on actual output format

        # Just check it doesn't explode - exact threshold TBD
        # This test documents import time for future reference
    ```

- [ ] **5.1.3** Reuse the smoke test from Phase 2.5

  - [ ] Copy the `test_audit_german_credit_simple_works` test
  - [ ] Or import it from test_audit_smoke.py

- [ ] **5.1.4** Run all performance tests
  - [ ] `cd packages && pytest tests/test_cli_performance.py -v -s`
  - [ ] Verify all pass with current changes
  - [ ] Record baseline times

### Step 5.2: CI Integration

- [ ] **5.2.1** Update CI configuration

  - [ ] Find CI config (`.github/workflows/*.yml` or similar)
  - [ ] Add performance test job:
    ```yaml
    - name: Performance tests
      run: |
        pytest tests/test_cli_performance.py -v
        pytest tests/test_audit_smoke.py -v
    ```
  - [ ] Set appropriate timeout (5 minutes should be plenty)

- [ ] **5.2.2** Consider pre-commit hook
  - [ ] Decide if performance check should be in pre-commit
  - [ ] If yes: add quick check to `scripts/pre-commit`
  - [ ] If no: document in CONTRIBUTING.md as manual check

---

## Phase 4: Infrastructure & Documentation (2 hours) 🏗️

**Deferred until after core fixes**

- [ ] **4.1.1** Create `packages/src/glassalpha/cli/lazy.py` (optional)

  - [ ] Add lazy command decorator utilities if pattern becomes common
  - [ ] Document usage

- [ ] **4.2.1** Add performance rules to `packages/README.md`

  - [ ] Add "CLI Performance" section
  - [ ] Document lazy loading patterns
  - [ ] Show examples of correct vs incorrect imports

- [ ] **4.2.2** Update architecture rules

  - [ ] Open main `README.md` or create `CONTRIBUTING.md`
  - [ ] Add CLI performance guidelines
  - [ ] Document target thresholds (<300ms help, <100ms version)

- [ ] **4.2.3** Update workspace rules
  - [ ] Add rule: "Never import ML libraries at module level in CLI code"
  - [ ] Add rule: "Never import ML libraries in `__init__.py` files"
  - [ ] Add rule: "Use `__getattr__` for backward-compatible lazy loading"

---

## Phase 6: Advanced Optimizations (Optional) ✨

**Only if numbers stall or extra polish needed**

- [ ] **6.1** Use TYPE_CHECKING guards

  - [ ] Find imports used only for type hints
  - [ ] Wrap in `if TYPE_CHECKING:` blocks
  - [ ] Use string literals for forward references if needed

- [ ] **6.2** Consider importlib.util.LazyLoader

  - [ ] Research if appropriate for remaining modules
  - [ ] Implement if beneficial
  - [ ] Document pattern

- [ ] **6.3** Profile other CLI commands

  - [ ] Measure: `time glassalpha audit --help`
  - [ ] Measure: `time glassalpha validate --help`
  - [ ] Measure: `time glassalpha list`
  - [ ] Fix any other slow commands

- [ ] **6.4** Optimize \_bootstrap_components()
  - [ ] Review what's imported during bootstrap
  - [ ] Defer anything not needed for --help
  - [ ] Add lazy loading for component discovery

---

## Final Validation ✅

- [ ] **V.1** Performance benchmarks

  - [ ] Measure: `time glassalpha --help` → Should be <300ms
  - [ ] Measure: `time glassalpha --version` → Should be <100ms
  - [ ] Measure: `time glassalpha audit --help` → Should be reasonable
  - [ ] Record all times for documentation

- [ ] **V.2** Functionality tests

  - [ ] Run full test suite: `pytest tests/ -v`
  - [ ] Test audit command: `glassalpha audit --config configs/german_credit_simple.yaml --out test.pdf`
  - [ ] Test dataset commands: `glassalpha datasets list`
  - [ ] Test validation: `glassalpha validate configs/quickstart.yaml`

- [ ] **V.3** Explainer registry validation

  - [ ] Verify no TypeError from is_compatible calls
  - [ ] Check logs for proper explainer selection
  - [ ] Confirm contract test passes

- [ ] **V.4** Documentation

  - [ ] Update CHANGELOG.md with performance improvements and explainer fix
  - [ ] Add note to README about fast CLI startup
  - [ ] Document any breaking changes (if any)

- [ ] **V.5** Code review
  - [ ] Review all changes for maintainability
  - [ ] Ensure lazy imports don't complicate code too much
  - [ ] Verify error messages are still clear
  - [ ] Check that import errors are handled gracefully

---

## Success Metrics 📊

Track these before/after:

| Metric                  | Before | Target | Actual   |
| ----------------------- | ------ | ------ | -------- |
| `--help` warm cache     | 635ms  | <300ms | **\_\_** |
| `--help` cold cache     | ~800ms | <800ms | **\_\_** |
| `--version` warm        | 635ms  | <100ms | **\_\_** |
| Explainer API           | BROKEN | FIXED  | **\_\_** |
| Audit smoke test        | N/A    | PASS   | **\_\_** |
| Full test suite         | PASS   | PASS   | **\_\_** |
| audit command           | CRASH  | Works  | **\_\_** |
| Explainer contract test | N/A    | PASS   | **\_\_** |

---

## Revised Order of Attack 🎯

**Critical path - do in this order:**

1. **Phase 2.5 NOW** (45-60 min)

   - Fix explainer API bug
   - Add contract tests
   - Add smoke test
   - **Outcome**: Audits work again

2. **Phase 1 + Phase 2** (2 hours)

   - Move sklearn import
   - Implement LazyGroup pattern
   - Fix datasets lazy loading
   - **Outcome**: --help drops to <200ms

3. **Phase 3 + Phase 5** (2 hours)

   - Audit and fix remaining imports
   - Add performance guards
   - CI integration
   - **Outcome**: Performance gains locked in, no regressions

4. **Phase 4 and 6** (Optional - 2-3 hours)
   - Only if numbers stall
   - Or if you want extra polish
   - Documentation and advanced optimizations

---

## Rollback Plan 🔄

If something breaks:

- [ ] **R.1** Git status check

  - [ ] Run: `git status` to see all changes
  - [ ] Run: `git diff` to review changes

- [ ] **R.2** Revert specific changes (in priority order)

  - [ ] Most critical: revert explainer API changes if tests fail
  - [ ] Next: revert `data/tabular.py` import change
  - [ ] Next: revert `datasets/__init__.py` changes
  - [ ] Last: revert CLI command registration changes

- [ ] **R.3** Test after revert
  - [ ] Verify tests pass
  - [ ] Verify CLI works
  - [ ] Verify audits complete

---

## Notes & Learnings 📝

Use this section to track issues, learnings, and improvements:

- Issue encountered:
- Solution applied:
- Time spent:
- Would do differently:

---

**Estimated Total Time**: 6-8 hours across critical path
**Minimum Viable**: Phase 2.5 + Phase 1 (2 hours) for working tool + 58% speedup
**Recommended**: Phases 2.5, 1, 2, 3, 5 (5-6 hours) for complete solution with guards
