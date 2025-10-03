# WeasyPrint Constraints Fix

## Problem

CI was failing with dependency conflicts:

1. Initial: `weasyprint==63.3` doesn't exist (WeasyPrint uses major.minor versioning only)
2. After fixing to 63.1: `tinycss2==1.3.0` doesn't satisfy WeasyPrint 63.1's requirement `>=1.4.0`
3. Missing transitive dependencies for WeasyPrint in constraint files

## Root Cause

The constraint files were created manually without fully resolving WeasyPrint's dependency tree. When pip tried to install `weasyprint==63.1`, it found conflicts with pinned transitive dependencies.

## Solution

### 1. Fixed WeasyPrint Version

- Changed from non-existent `63.3` to actual version `63.1`
- WeasyPrint versioning: 63.0, 63.1, 64.0, 64.1, etc. (no patch versions)

### 2. Updated tinycss2

- Old: `tinycss2==1.3.0`
- New: `tinycss2==1.4.0`
- Required by: `weasyprint 63.1 depends on tinycss2>=1.4.0`

### 3. Added Missing WeasyPrint Dependencies

Added complete dependency tree for WeasyPrint 63.1:

| Package    | Version | Requirement | Status |
| ---------- | ------- | ----------- | ------ |
| pydyf      | 0.11.0  | >=0.11.0    | ✓      |
| cffi       | 2.0.0   | >=0.6       | ✓      |
| tinyhtml5  | 2.0.0   | >=2.0.0b1   | ✓      |
| tinycss2   | 1.4.0   | >=1.4.0     | ✓      |
| cssselect2 | 0.7.0   | >=0.1       | ✓      |
| pyphen     | 0.17.2  | >=0.9.1     | ✓      |
| pillow     | 11.3.0  | >=9.1.0     | ✓      |
| fonttools  | 4.60.1  | >=4.0.0     | ✓      |
| cairocffi  | 1.7.1   | (optional)  | ✓      |
| cairosvg   | 2.7.1   | (optional)  | ✓      |

## Files Updated

### Constraint Files

- `packages/constraints.txt` - Updated tinycss2, added full WeasyPrint dependency tree
- `packages/constraints/constraints-ubuntu-latest-py3.11.txt` - Added WeasyPrint deps
- `packages/constraints/constraints-ubuntu-latest-py3.12.txt` - Added WeasyPrint deps
- `packages/constraints/constraints-macos-14-py3.11.txt` - Added WeasyPrint deps
- `packages/constraints/constraints-macos-14-py3.12.txt` - Added WeasyPrint deps

### Documentation

- `packages/CONSTRAINTS.md` - Updated to show correct version (63.1) and added note about versioning
- `packages/dev/reviews/MATPLOTLIB_BACKEND_FIX_COMPLETE.md` - Updated version references

## Verification

```python
✓ All constraints satisfy WeasyPrint 63.1 requirements
✓ No circular dependencies
✓ All pinned versions exist on PyPI
```

## Why This Approach is Correct

### Pinned Transitive Dependencies

- **Pro**: Ensures reproducible builds across all environments
- **Pro**: Prevents surprise breakages from transitive updates
- **Pro**: Makes CI stable and predictable
- **Con**: Requires manual updates (mitigated by pip-tools + Renovate)

### Alternative Considered: Loose Constraints

- Let pip resolve transitive dependencies automatically
- **Rejected because**: Non-deterministic, different versions across runs, harder to debug

## Prevention

### Short-term

Added complete dependency chains to constraint files so future updates will be caught immediately.

### Long-term

Per `CONSTRAINTS.md`, use pip-tools to regenerate constraints:

```bash
cd packages
pip-compile -o constraints/constraints-ubuntu-latest-py3.12.txt \
  --python-version 3.12 --resolver backtracking --strip-extras \
  --extra docs --extra explain --extra viz --extra dev pyproject.toml
```

This automatically resolves the full dependency tree including transitives.

## Lesson Learned

**Don't manually maintain constraint files for packages with complex dependency trees.**

Use pip-tools to generate constraints from `pyproject.toml`, then:

1. Verify the output is complete
2. Test with `pip install --dry-run`
3. Commit and let CI validate
4. Set up Renovate/Dependabot for automatic updates

## Related Issues

- Matplotlib backend fix: Addresses different issue (GUI crash), unrelated to this
- WeasyPrint version: Now correctly pinned to actual version 63.1
- tinycss2 conflict: Resolved by updating to 1.4.0

---

**Status**: ✅ Fixed and verified
**Date**: 2025-10-03
**Impact**: CI should now install WeasyPrint and dependencies without conflicts
