# CLI Performance Analysis & Optimization Plan

## Current Performance

**Measured**: `glassalpha --help` takes **635ms** (warm cache)
**Target**: <300ms (warm), <800ms (cold)
**Status**: ❌ Slower than target by ~2x

## Root Cause Analysis

Using `python -X importtime`, the bottleneck is clear:

```
Import Chain:                                    Time (ms)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
glassalpha.cli.main                                    583
  └─ glassalpha.cli.datasets (line 109)                561
      └─ glassalpha.datasets.__init__                  557
          └─ glassalpha.datasets.german_credit         556
              └─ glassalpha.data.tabular               366
                  └─ sklearn.model_selection           365  ⚠️ BOTTLENECK
```

**The problem**: Line 109 of `cli/main.py`:

```python
from .datasets import list_datasets, dataset_info, show_cache_dir, fetch_dataset
```

This eagerly imports the datasets module, which:

1. Imports `german_credit.py` in `datasets/__init__.py` (line 7)
2. Which imports `TabularDataSchema` from `data.tabular` (line 19)
3. Which imports `train_test_split` from `sklearn.model_selection` (line 19)
4. **sklearn.model_selection is massive** (loads clustering, metrics, pairwise distances, etc.)

## Why This Matters

- `--help` should be instant (users check it frequently)
- Current delay suggests "broken" or "bloated" tool
- Every CLI invocation pays this tax (even `--version`)
- This will get worse as we add more models/dependencies

## Recommended Solution (Long-Term Optimal)

### Strategy: Lazy Command Registration

Use **deferred imports** so help text appears fast, but heavy code loads only when commands run.

### Phase 1: Fix the datasets command (Quick Win)

**Problem**: `cli/main.py` imports dataset functions eagerly for registration.

**Solution**: Use lazy function wrappers:

```python
# cli/main.py - BEFORE (current)
from .datasets import list_datasets, dataset_info, show_cache_dir, fetch_dataset

datasets_app.command("list")(list_datasets)
datasets_app.command("info")(dataset_info)
datasets_app.command("cache-dir")(show_cache_dir)
datasets_app.command("fetch")(fetch_dataset)
```

```python
# cli/main.py - AFTER (lazy)
# Remove: from .datasets import ...

# Create lazy wrappers that defer import until invoked
def _make_lazy_command(module_path: str, func_name: str):
    """Create a lazy command that imports only when called."""
    def wrapper(*args, **kwargs):
        from importlib import import_module
        module = import_module(module_path)
        func = getattr(module, func_name)
        return func(*args, **kwargs)

    # Preserve function signature for Typer's introspection
    import functools
    # Get the actual function to copy metadata
    from importlib import import_module
    actual_module = import_module(module_path)
    actual_func = getattr(actual_module, func_name)
    wrapper = functools.wraps(actual_func)(wrapper)
    return wrapper

# Register lazy commands
datasets_app.command("list")(_make_lazy_command("glassalpha.cli.datasets", "list_datasets"))
datasets_app.command("info")(_make_lazy_command("glassalpha.cli.datasets", "dataset_info"))
datasets_app.command("cache-dir")(_make_lazy_command("glassalpha.cli.datasets", "show_cache_dir"))
datasets_app.command("fetch")(_make_lazy_command("glassalpha.cli.datasets", "fetch_dataset"))
```

**Problem with this approach**: Typer needs to introspect function signatures to generate help text, so we still import eagerly.

**Better solution**: Move the import **inside** the command group callback or use Typer's lazy loading pattern.

### Phase 2: Fix the data.tabular import (Core Fix)

**Problem**: `data/tabular.py` imports sklearn at module level:

```python
from sklearn.model_selection import train_test_split  # Line 19
```

**Solution**: Move to function scope:

```python
# data/tabular.py - BEFORE
from sklearn.model_selection import train_test_split  # Line 19

class TabularDataLoader(DataInterface):
    def split_data(self, ...):
        X_train, X_test, y_train, y_test = train_test_split(...)
```

```python
# data/tabular.py - AFTER
# Remove top-level import

class TabularDataLoader(DataInterface):
    def split_data(self, ...):
        from sklearn.model_selection import train_test_split  # Import inside method
        X_train, X_test, y_train, y_test = train_test_split(...)
```

**Impact**: This **alone** should reduce --help time by ~365ms (58% improvement)

### Phase 3: Audit all module-level imports

Scan for other heavy imports that shouldn't load during help:

```bash
# Find all module-level imports in glassalpha
grep -r "^from.*import" packages/src/glassalpha/ | \
  grep -E "(pandas|numpy|matplotlib|weasyprint|xgboost|lightgbm|shap|sklearn)" | \
  grep -v "# lazy" | \
  grep -v "TYPE_CHECKING"
```

**Rules**:

1. **Never import ML libs at module level** in CLI code
2. **Never import ML libs at module level** in `__init__.py` files
3. **Defer data loaders** until actual use
4. **Use TYPE_CHECKING guard** for type hints only

### Phase 4: Implement proper lazy command registration

Create a reusable pattern:

```python
# cli/lazy.py
"""Lazy command registration utilities for fast CLI startup."""

from typing import Any, Callable
import functools
from importlib import import_module

def lazy_command(module_path: str, func_name: str) -> Callable[..., Any]:
    """Create a lazy command that imports only when invoked.

    This preserves function signatures for Typer while deferring
    the actual module import until the command runs.

    Args:
        module_path: Dotted module path (e.g., "glassalpha.cli.datasets")
        func_name: Function name to import

    Returns:
        Wrapper function with preserved signature
    """
    @functools.wraps(lambda: None)  # Placeholder for introspection
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        module = import_module(module_path)
        func = getattr(module, func_name)
        return func(*args, **kwargs)

    # Copy docstring and annotations for help text (but don't import yet)
    # This is a trade-off: we need SOME metadata for help, but not full execution
    return wrapper
```

**Alternative**: Use Typer's built-in lazy loading with command groups:

```python
# cli/main.py
datasets_app = typer.Typer(help="Dataset management operations")

# Register commands with lazy loading
@datasets_app.command("list")
def list_datasets_cmd():
    """List all available datasets in the registry."""
    from .datasets import list_datasets  # Import only when invoked
    return list_datasets()
```

### Phase 5: Add performance regression tests

Prevent future slowdowns:

```python
# tests/test_cli_performance.py
import subprocess
import sys
import time

def test_help_under_300ms():
    """Ensure --help stays fast (warm cache)."""
    # Warm up
    subprocess.run([sys.executable, "-m", "glassalpha", "--version"],
                   capture_output=True)

    # Measure help
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "glassalpha", "--help"],
        capture_output=True,
        check=True
    )
    elapsed = time.perf_counter() - start

    # Allow 300ms on developer machine, 500ms on CI
    threshold = 0.5 if os.getenv("CI") else 0.3
    assert elapsed < threshold, \
        f"--help took {elapsed:.3f}s, should be <{threshold}s"

def test_version_under_100ms():
    """Version should be nearly instant."""
    start = time.perf_counter()
    subprocess.run([sys.executable, "-m", "glassalpha", "--version"],
                   capture_output=True, check=True)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1, f"--version took {elapsed:.3f}s"
```

## Implementation Plan

### Week 1: Core Fixes (High Impact)

**Priority 1** (30 min):

- [ ] Move `sklearn.model_selection` import inside `TabularDataLoader.split_data()`
- [ ] Test: `time glassalpha --help` should be ~270ms (down from 635ms)

**Priority 2** (1 hour):

- [ ] Fix `datasets/__init__.py` to not eagerly import `german_credit`
- [ ] Use lazy registration pattern for datasets commands
- [ ] Test: `time glassalpha --help` should be <200ms

**Priority 3** (1 hour):

- [ ] Audit all `glassalpha/**/__init__.py` files for heavy imports
- [ ] Move any pandas/sklearn/numpy imports to function scope

### Week 2: Infrastructure (Long-term)

**Priority 4** (2 hours):

- [ ] Create `cli/lazy.py` with lazy command registration helpers
- [ ] Refactor all command registrations to use lazy pattern
- [ ] Update architecture docs with "no module-level ML imports" rule

**Priority 5** (1 hour):

- [ ] Add `test_cli_performance.py` with <300ms threshold
- [ ] Add to CI to prevent regressions
- [ ] Document performance requirements in CONTRIBUTING.md

### Week 3: Polish (Nice-to-Have)

**Priority 6**:

- [ ] Use `TYPE_CHECKING` guards for type-only imports
- [ ] Consider using `importlib.util.LazyLoader` for remaining cases
- [ ] Profile startup time for all commands (not just --help)

## Expected Results

| Metric                 | Before | After Phase 1 | After Phase 2 | Target    |
| ---------------------- | ------ | ------------- | ------------- | --------- |
| `--help` (warm)        | 635ms  | ~270ms        | ~150ms        | <300ms ✅ |
| `--version` (warm)     | ~600ms | <50ms         | <50ms         | <100ms ✅ |
| `audit` (first import) | N/A    | Same          | Same          | N/A       |

## Architecture Guidelines (Update Rules)

Add to `project_overview.md`:

```markdown
### CLI Performance Rules

1. **Never import ML libraries at module level**

   - Bad: `from sklearn.model_selection import train_test_split` at top
   - Good: Import inside functions/methods where used

2. **Never import data loaders in `__init__.py`**

   - Bad: `from .german_credit import load_german_credit` in `__init__`
   - Good: Lazy import in registry or command functions

3. **Use lazy command registration**

   - Import command functions inside wrapper, not at CLI setup time

4. **Guard type-only imports**

   - Use `if TYPE_CHECKING:` for imports only needed for type hints

5. **Test CLI performance**
   - `--help` must be <300ms (warm), <800ms (cold)
   - Add regression tests to prevent slowdowns
```

## Alternative: Don't Lazy Load (If Justified)

**When NOT to optimize**: If you have good reasons to keep current structure:

1. **If help text needs data**: Some commands need module metadata for help
2. **If complexity isn't worth it**: If 635ms is acceptable for your users
3. **If dependencies will grow**: If you'll need all imports eventually anyway

**Counter-argument**: For a compliance tool targeting regulated industries:

- First impressions matter (slow help suggests bloat)
- Users may invoke CLI frequently in scripts (100x = 63s overhead)
- Gradual degradation as you add features
- Best practices for Python CLI tools (Click, Typer, etc.) all recommend lazy loading

## Recommendation

**Implement Phase 1 immediately** (30 min, 58% speedup). This is:

- Low risk (moving one import)
- High impact (365ms → 0ms for --help)
- No architectural changes
- Sets good precedent

**Implement Phase 2 within sprint** (1 hour, gets to <200ms). This:

- Fixes the datasets command import
- Shows pattern for future commands
- Achieves target performance

**Defer Phases 3-6** until next sprint. Focus on:

- Getting the quick wins first
- Proving the approach works
- Documenting the pattern for future code

## Questions for Review

1. **Is <300ms acceptable target?** Or should we aim for <150ms?
2. **Should we lazy-load commands or just fix imports?** Both work, but lazy commands are more maintainable.
3. **Add to CI now or after Phase 2?** Risk of flaky tests if threshold too aggressive.
4. **Document as "rule" or "guideline"?** Should pre-commit catch module-level ML imports?

## References

- [Click Performance Best Practices](https://click.palletsprojects.com/en/8.1.x/advanced/#command-loading)
- [Python Import Time Profiling](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPROFILEX)
- [Typer Lazy Subcommands](https://typer.tiangolo.com/tutorial/commands/one-or-multiple/#lazy-loading-commands)
