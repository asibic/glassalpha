"""JSON error output for CI/CD integration.

This module provides machine-readable error output for automation and
continuous integration pipelines.
"""

import json
import sys
from datetime import UTC, datetime
from typing import Any

from .exit_codes import ExitCode


class JSONErrorOutput:
    """Machine-readable JSON error output for CI/CD.

    Provides structured error information that can be parsed by automation
    tools, CI/CD pipelines, and monitoring systems.
    """

    @staticmethod
    def format_error(
        exit_code: int,
        error_type: str,
        message: str,
        details: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Format error as structured JSON.

        Args:
            exit_code: Numeric exit code
            error_type: Error category (CONFIG, DATA, MODEL, SYSTEM, VALIDATION, COMPONENT)
            message: Human-readable error message
            details: Additional error details
            context: Contextual information (file paths, line numbers, etc.)

        Returns:
            Structured error dictionary

        """
        return {
            "status": "error",
            "exit_code": exit_code,
            "exit_code_name": ExitCode(exit_code).name if exit_code in [e.value for e in ExitCode] else "UNKNOWN",
            "error": {
                "type": error_type,
                "message": message,
                "details": details or {},
                "context": context or {},
            },
            "timestamp": datetime.now(UTC).isoformat(),
            "version": "1.0",  # JSON schema version
        }

    @staticmethod
    def output_error(
        exit_code: int,
        error_type: str,
        message: str,
        details: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        file: Any = None,
    ) -> None:
        """Output error as JSON to stderr.

        Args:
            exit_code: Numeric exit code
            error_type: Error category
            message: Human-readable error message
            details: Additional error details
            context: Contextual information
            file: Output file handle (default: stderr)

        """
        error_data = JSONErrorOutput.format_error(
            exit_code=exit_code,
            error_type=error_type,
            message=message,
            details=details,
            context=context,
        )

        output_file = file or sys.stderr
        json.dump(error_data, output_file, indent=2)
        output_file.write("\n")
        output_file.flush()

    @staticmethod
    def format_validation_errors(
        errors: list[dict[str, Any]],
        config_path: str | None = None,
    ) -> dict[str, Any]:
        """Format multiple validation errors.

        Args:
            errors: List of validation error dictionaries
            config_path: Path to configuration file being validated

        Returns:
            Structured validation error response

        """
        return {
            "status": "validation_failed",
            "exit_code": ExitCode.VALIDATION_ERROR,
            "exit_code_name": "VALIDATION_ERROR",
            "validation": {
                "passed": False,
                "error_count": len(errors),
                "errors": errors,
                "config_path": config_path,
            },
            "timestamp": datetime.now(UTC).isoformat(),
            "version": "1.0",
        }

    @staticmethod
    def format_success(
        message: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Format success response.

        Args:
            message: Success message
            data: Additional data to include

        Returns:
            Structured success response

        """
        return {
            "status": "success",
            "exit_code": ExitCode.SUCCESS,
            "exit_code_name": "SUCCESS",
            "message": message,
            "data": data or {},
            "timestamp": datetime.now(UTC).isoformat(),
            "version": "1.0",
        }

    @staticmethod
    def output_success(
        message: str,
        data: dict[str, Any] | None = None,
        file: Any = None,
    ) -> None:
        """Output success as JSON to stdout.

        Args:
            message: Success message
            data: Additional data to include
            file: Output file handle (default: stdout)

        """
        success_data = JSONErrorOutput.format_success(message=message, data=data)

        output_file = file or sys.stdout
        json.dump(success_data, output_file, indent=2)
        output_file.write("\n")
        output_file.flush()


def json_error_handler(
    exit_code: int,
    error_type: str,
    message: str,
    details: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """Convenience function for outputting JSON errors.

    Args:
        exit_code: Numeric exit code
        error_type: Error category
        message: Human-readable error message
        details: Additional error details
        context: Contextual information

    """
    JSONErrorOutput.output_error(
        exit_code=exit_code,
        error_type=error_type,
        message=message,
        details=details,
        context=context,
    )


def should_use_json_output() -> bool:
    """Check if JSON output mode is enabled.

    Checks for:
    - GLASSALPHA_JSON_ERRORS environment variable
    - CI environment variables (CI, GITHUB_ACTIONS, etc.)

    Returns:
        True if JSON output should be used

    """
    import os

    # Explicit JSON errors flag
    if os.environ.get("GLASSALPHA_JSON_ERRORS", "").lower() in ("1", "true", "yes"):
        return True

    # Auto-enable in CI environments
    ci_vars = ["GITHUB_ACTIONS", "GITLAB_CI", "CIRCLECI", "JENKINS_HOME", "TRAVIS"]
    if any(os.environ.get(var) for var in ci_vars):
        return True

    return False
