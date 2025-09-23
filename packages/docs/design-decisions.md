# Glass Alpha Design Decisions

## CLI Error Handling (September 2024)

**Decision**: Use `raise typer.Exit(1) from None` in CLI exception handlers

**Rationale**:
- CLI applications should provide clean, user-friendly error messages
- End users don't need to see Python stack traces for common errors like "file not found"
- The `from None` explicitly suppresses the chain while satisfying linter requirements
- Verbose mode (`--verbose`) still provides full debugging information when needed

**Alternatives Considered**:
- Standard exception chaining (`from e`): Shows technical Python details to end users
- Bare `raise typer.Exit(1)`: Works but triggers B904 linter warnings

**Examples**:
```python
# GOOD - Clean user experience
except FileNotFoundError as e:
    typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1) from None  # User sees: "Error: File not found: config.yaml"

# BAD - Scary for users
except FileNotFoundError as e:
    typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1) from e  # User sees: Full Python traceback + file paths
```

**Status**: Active - Do not "fix" this pattern

**Files Affected**: `src/glassalpha/cli/*.py`

---

## Typer Function Call Defaults (September 2024)

**Decision**: Use function calls in Typer argument defaults (triggers B008 lint rule)

**Rationale**:
- This is the documented Typer pattern for CLI argument definition
- Typer requires function calls to properly configure click options
- Alternative approaches don't provide the same functionality

**Example**:
```python
# CORRECT - Typer's documented pattern
def command(
    config: Path = typer.Option(  # B008 warning, but necessary
        ...,
        "--config",
        help="Configuration file",
        exists=True  # This validation requires the function call
    )
):
```

**Status**: Active - This is the correct Typer usage

**Files Affected**: `src/glassalpha/cli/*.py`

---

## Test File Import Patterns (September 2024)

**Decision**: Allow module-level imports after setup code in test files (E402)

**Rationale**:
- Tests need to mock dependencies (pandas, numpy) before importing our modules
- This is a common and necessary testing pattern
- Moving imports to top would break the mocking

**Example**:
```python
# Mock dependencies first
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Then import our code that depends on them
from glassalpha.core import ModelRegistry  # E402 warning, but necessary
```

**Status**: Active - Required for proper dependency mocking

**Files Affected**: `tests/*.py`

---

## Configuration Validation Import Check (September 2024)

**Decision**: Import tensorflow in `strict.py` to detect its presence (F401)

**Rationale**:
- The import is used to detect if tensorflow is in the environment
- This affects reproducibility warnings in strict mode
- The import is intentional, even though the module isn't directly used

**Example**:
```python
try:
    import tensorflow  # F401 warning, but this IS the usage
    issues.append("TensorFlow detected - may introduce randomness")
except ImportError:
    pass  # TF not installed, no issue
```

**Status**: Active - This IS the intended usage

**Files Affected**: `src/glassalpha/config/strict.py`

---

## Linter Configuration Philosophy

**Principle**: Configure linters to understand architectural intent, not blindly follow rules.

**Implementation**: Use per-file ignores in `pyproject.toml` to disable specific rules where they conflict with intentional design patterns.

**Key Files**:
- `pyproject.toml` - Per-file linter configuration
- `src/glassalpha/cli/*.py` - Strategic comments explaining design choices
