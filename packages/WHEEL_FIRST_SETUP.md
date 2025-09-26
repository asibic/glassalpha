# Wheel-First Anti-Thrashing Setup

## 🎯 **Problem Solved**
Prevents CI thrashing cycle where local fixes don't predict CI success, causing endless debug loops.

## 🔧 **Solution: Wheel-First Validation**

### Core Principle
**Test the wheel CI actually uses, not the source CI ignores.**

```bash
# OLD (Thrashing): Edit → Push → CI fails → Repeat
# NEW (Locked in): Edit → `make smoke` → Fix locally → Push (with confidence)
```

## 📋 **Files Added**

### 1. **scripts/wheel_smoke.sh** - The Key Script
- Builds wheel exactly like CI
- Validates all 4 critical contracts directly in wheel contents
- No dependency issues - tests wheel structure
- **30 second feedback** vs 5+ minute CI cycle

### 2. **Git Hooks** - Smart Automatic Testing
- **`scripts/pre-commit`** - Runs smoke test ONLY when fragile contract areas change
- **`scripts/pre-push`** - Always runs smoke test before every push
- **Blocks bad commits/pushes** that would cause CI thrashing
- Perfect for solo development - catches regressions without slowdown

### 3. **tests/contracts/** - Regression Guards
- `test_contracts_smoke.py` - Meta-tests validate wheel contents
- `test_logger_exact.py` - Specific tripwire for the chronic logger failure
- Tests installed package, not source

### 4. **src/glassalpha/constants.py** - Centralized Fragile Strings
- All error messages, log templates, manifest structure
- Prevents typo drift between code and tests
- Single source of truth for contract strings

### 5. **Makefile** - Easy Access
- `make smoke` - Run wheel validation
- `make check` - Full pre-push checklist
- `make help` - Show all commands

### 6. **Optional Packaging Assert**
- In `report/__init__.py` with `GLASSALPHA_ASSERT_PACKAGING=1`
- Catches missing templates immediately in CI
- Off by default, enable in CI once

## 🚀 **Usage**

### One-Time Setup
```bash
# Install both git hooks (run once)
make hooks
```

### Daily Development
```bash
# Make changes, commit normally
git add .
git commit -m "feat: whatever"
# → pre-commit hook runs smoke test ONLY if you touched fragile areas

# Push when ready
git push
# → pre-push hook ALWAYS runs smoke test (final safety net)
```

### Manual Testing (Optional)
```bash
# Run smoke test manually anytime
make smoke
```

### Debugging CI (Rare)
```bash
# Only if local wheel is green but CI is red
make smoke  # ✅ All contracts validated
git push    # CI fails anyway

# Now send CI logs - we know it's environment, not code
```

## ✅ **4 Critical Contracts Validated**

1. **Logger Format**: f-string single argument (not printf style)
2. **Template Packaging**: `standard_audit.html` in installed wheel
3. **Model Training**: Simplified fit logic without complex conditions
4. **Save/Load Symmetry**: Perfect roundtrip in LogisticRegressionWrapper

## 🔒 **Drift Prevention**

### Smart Git Hooks
- **Pre-commit**: Only runs when fragile areas change (fast, focused)
- **Pre-push**: Always runs (final safety net, blocks bad pushes)
- **Automatic enforcement** - can't commit/push broken contracts
- Clear fix instructions when smoke test fails

### Centralized Contracts
- Error messages in `constants.py` prevent typos
- Manifest structure helper ensures exact E2E format
- Template names centralized for packaging tests

### Contract Guard Tests
- `tests/contracts/` prevent regression of chronic failures
- Test installed package (matches CI exactly)
- Specific tripwire for logger issue that caused most thrashing

## 🎉 **Benefits Locked In**

- **No more CI thrashing** - Contracts validated locally first
- **Fast feedback** - 30 second smoke test vs 5+ minute CI
- **Exact testing** - Tests the wheel CI uses, not source
- **Automatic prevention** - Pre-push hook blocks bad pushes
- **Clear diagnostics** - Know exactly which contract failed
- **Drift-proof** - Centralized strings prevent typo regression

## 📝 **When to Send CI Logs**

**Only after:**
1. `make smoke` is ✅ green locally
2. Full wheel install passes locally
3. CI still fails

**Then:** Send only the failing test log - we know it's environment differences.

## 🔧 **Future CI Setup**

Use `ci-template.yml` which implements wheel-first in CI:
- Build wheel first (never editable install)
- Install from wheel (matches local)
- Run contract guards first (fast feedback)
- Optional packaging assertion with env var

**Critical:** CI must match local wheel-first approach or we'll drift again.
