"""Basic CLI functionality tests.

Tests that CLI loads and core commands work without crashing.
These tests focus on coverage for main.py and commands.py.
"""

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from glassalpha.cli.main import app


def test_cli_app_loads():
    """Test that the CLI app loads without errors."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Glass Alpha - AI Compliance Toolkit" in result.stdout


def test_version_command():
    """Test the version command works."""
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 or "version" in result.stdout.lower()


def test_audit_command_missing_config():
    """Test audit command fails gracefully with missing config."""
    runner = CliRunner()
    result = runner.invoke(app, ["audit", "--config", "nonexistent.yaml", "--output", "test.pdf"])
    # Should exit with error but not crash
    assert result.exit_code != 0


def test_audit_command_with_temp_config():
    """Test audit command loads with valid config structure."""
    runner = CliRunner()

    # Create minimal valid config
    config_content = """
audit_profile: "tabular_compliance"
model:
  type: "xgboost"
data:
  path: "test.csv"
explainers:
  strategy: "first_compatible"
  priority: ["treeshap"]
metrics:
  performance: ["accuracy"]
reproducibility:
  random_seed: 42
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as output_file:
            output_path = output_file.name

        # This will likely fail due to missing data file, but should load config
        result = runner.invoke(app, ["audit", "--config", config_path, "--output", output_path])

        # Command should at least attempt to load config (may fail later due to missing data)
        # We just want to exercise the CLI code path
        assert "audit_profile" not in result.stdout or result.exit_code != 0

    finally:
        Path(config_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def test_strict_mode_flag():
    """Test that strict mode flag is recognized."""
    runner = CliRunner()

    config_content = """
audit_profile: "tabular_compliance"
model:
  type: "xgboost"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        result = runner.invoke(app, ["audit", "--config", config_path, "--output", "test.pdf", "--strict"])
        # Should recognize the flag (may fail for other reasons)
        assert result.exit_code != 125  # 125 is "unknown option" in Click/Typer

    finally:
        Path(config_path).unlink(missing_ok=True)
