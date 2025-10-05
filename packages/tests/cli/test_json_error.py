"""Tests for JSON error output."""

import json
from io import StringIO

from glassalpha.cli.exit_codes import ExitCode
from glassalpha.cli.json_error import (
    JSONErrorOutput,
    json_error_handler,
    should_use_json_output,
)


class TestJSONErrorOutput:
    """Tests for JSONErrorOutput class."""

    def test_format_error_basic(self):
        """Should format basic error as JSON."""
        result = JSONErrorOutput.format_error(
            exit_code=ExitCode.USER_ERROR,
            error_type="CONFIG",
            message="Invalid configuration",
        )

        assert result["status"] == "error"
        assert result["exit_code"] == ExitCode.USER_ERROR
        assert result["exit_code_name"] == "USER_ERROR"
        assert result["error"]["type"] == "CONFIG"
        assert result["error"]["message"] == "Invalid configuration"
        assert result["error"]["details"] == {}
        assert result["error"]["context"] == {}
        assert "timestamp" in result
        assert result["version"] == "1.0"

    def test_format_error_with_details(self):
        """Should include details and context."""
        result = JSONErrorOutput.format_error(
            exit_code=ExitCode.VALIDATION_ERROR,
            error_type="DATA",
            message="Missing column",
            details={"missing_columns": ["age", "income"]},
            context={"file": "data.csv", "line": 42},
        )

        assert result["error"]["details"]["missing_columns"] == ["age", "income"]
        assert result["error"]["context"]["file"] == "data.csv"
        assert result["error"]["context"]["line"] == 42

    def test_output_error(self):
        """Should output error to stderr as JSON."""
        output = StringIO()

        JSONErrorOutput.output_error(
            exit_code=ExitCode.SYSTEM_ERROR,
            error_type="SYSTEM",
            message="Permission denied",
            file=output,
        )

        output.seek(0)
        result = json.loads(output.read())

        assert result["status"] == "error"
        assert result["exit_code"] == ExitCode.SYSTEM_ERROR
        assert result["error"]["message"] == "Permission denied"

    def test_format_validation_errors(self):
        """Should format multiple validation errors."""
        errors = [
            {"field": "model.type", "message": "Invalid model type"},
            {"field": "data.path", "message": "File not found"},
        ]

        result = JSONErrorOutput.format_validation_errors(
            errors=errors,
            config_path="audit.yaml",
        )

        assert result["status"] == "validation_failed"
        assert result["exit_code"] == ExitCode.VALIDATION_ERROR
        assert result["validation"]["passed"] is False
        assert result["validation"]["error_count"] == 2
        assert result["validation"]["errors"] == errors
        assert result["validation"]["config_path"] == "audit.yaml"

    def test_format_success(self):
        """Should format success response."""
        result = JSONErrorOutput.format_success(
            message="Audit completed",
            data={"output": "report.html"},
        )

        assert result["status"] == "success"
        assert result["exit_code"] == ExitCode.SUCCESS
        assert result["exit_code_name"] == "SUCCESS"
        assert result["message"] == "Audit completed"
        assert result["data"]["output"] == "report.html"

    def test_output_success(self):
        """Should output success to stdout as JSON."""
        output = StringIO()

        JSONErrorOutput.output_success(
            message="Validation passed",
            data={"config": "valid"},
            file=output,
        )

        output.seek(0)
        result = json.loads(output.read())

        assert result["status"] == "success"
        assert result["message"] == "Validation passed"


class TestJSONErrorHandler:
    """Tests for json_error_handler convenience function."""

    def test_json_error_handler(self):
        """Should output error using convenience function."""
        output = StringIO()

        # Temporarily replace stderr
        import sys

        old_stderr = sys.stderr
        sys.stderr = output

        try:
            json_error_handler(
                exit_code=ExitCode.USER_ERROR,
                error_type="CONFIG",
                message="Test error",
            )
        finally:
            sys.stderr = old_stderr

        output.seek(0)
        result = json.loads(output.read())

        assert result["error"]["message"] == "Test error"


class TestShouldUseJSONOutput:
    """Tests for should_use_json_output function."""

    def test_explicit_flag(self, monkeypatch):
        """Should return True if GLASSALPHA_JSON_ERRORS=1."""
        monkeypatch.setenv("GLASSALPHA_JSON_ERRORS", "1")
        assert should_use_json_output() is True

    def test_explicit_flag_true(self, monkeypatch):
        """Should return True if GLASSALPHA_JSON_ERRORS=true."""
        monkeypatch.setenv("GLASSALPHA_JSON_ERRORS", "true")
        assert should_use_json_output() is True

    def test_github_actions(self, monkeypatch):
        """Should auto-enable in GitHub Actions."""
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        assert should_use_json_output() is True

    def test_gitlab_ci(self, monkeypatch):
        """Should auto-enable in GitLab CI."""
        monkeypatch.setenv("GITLAB_CI", "true")
        assert should_use_json_output() is True

    def test_circleci(self, monkeypatch):
        """Should auto-enable in CircleCI."""
        monkeypatch.setenv("CIRCLECI", "true")
        assert should_use_json_output() is True

    def test_jenkins(self, monkeypatch):
        """Should auto-enable in Jenkins."""
        monkeypatch.setenv("JENKINS_HOME", "/var/jenkins")
        assert should_use_json_output() is True

    def test_travis(self, monkeypatch):
        """Should auto-enable in Travis CI."""
        monkeypatch.setenv("TRAVIS", "true")
        assert should_use_json_output() is True

    def test_no_ci(self, monkeypatch):
        """Should return False if not in CI."""
        # Clear all CI env vars
        for var in ["GLASSALPHA_JSON_ERRORS", "GITHUB_ACTIONS", "GITLAB_CI", "CIRCLECI", "JENKINS_HOME", "TRAVIS"]:
            monkeypatch.delenv(var, raising=False)

        assert should_use_json_output() is False
