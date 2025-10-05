"""Unified error formatting for consistent CLI error messages.

This module provides a standardized way to format error messages across all
CLI commands, ensuring users always get clear, actionable information when
something goes wrong.

Error Message Structure:
    ERROR [TYPE]: Brief one-line description
      What: Detailed explanation of what happened
      Why: Root cause or reason for the error
      Fix: How to resolve the issue (actionable steps)

Example:
        specific command or action to fix

Examples:
    >>> from .error_formatter import ErrorFormatter, ErrorType
    >>> formatter = ErrorFormatter()
    >>> error = formatter.format_error(
    ...     error_type=ErrorType.CONFIG,
    ...     message="Configuration file not found",
    ...     what="The specified configuration file does not exist",
    ...     why="The path may be incorrect or the file hasn't been created yet",
    ...     fix="Check the path and create the file if needed",
    ...     example="glassalpha init --output config.yaml"
    ... )
    >>> print(error)
    ERROR [CONFIG]: Configuration file not found
      What: The specified configuration file does not exist
      Why: The path may be incorrect or the file hasn't been created yet
      Fix: Check the path and create the file if needed

Example:
        glassalpha init --output config.yaml

"""

from enum import Enum

import typer


class ErrorType(str, Enum):
    """Types of errors that can occur in the CLI.

    These correspond to the exit codes but provide more semantic meaning
    for error message formatting.

    Attributes:
        CONFIG: Configuration-related errors (files, syntax, validation)
        DATA: Data-related errors (missing files, schema issues, corrupt data)
        MODEL: Model-related errors (not found, incompatible, failed to load)
        SYSTEM: System-level errors (permissions, resources, environment)
        VALIDATION: Validation failures (strict mode, compliance requirements)
        COMPONENT: Component errors (missing explainer, unavailable metric)

    """

    CONFIG = "CONFIG"
    DATA = "DATA"
    MODEL = "MODEL"
    SYSTEM = "SYSTEM"
    VALIDATION = "VALIDATION"
    COMPONENT = "COMPONENT"


class ErrorFormatter:
    """Formats error messages in a consistent, user-friendly way.

    The formatter ensures all errors follow the same structure:
    - Brief one-line summary
    - Detailed explanation (What happened)
    - Root cause (Why it happened)
    - Resolution steps (How to fix)
    - Example command (when applicable)

    This makes errors self-diagnosable and reduces support burden.

    Attributes:
        use_color: Whether to use colored output (default: True)

    Examples:
        >>> formatter = ErrorFormatter()
        >>> error = formatter.format_config_error(
        ...     message="Invalid YAML syntax",
        ...     what="The configuration file contains invalid YAML",
        ...     why="Missing colon after 'model' on line 5",
        ...     fix="Add a colon after 'model:' and ensure proper indentation"
        ... )
        >>> formatter.print_error(error)
        ERROR [CONFIG]: Invalid YAML syntax
          What: The configuration file contains invalid YAML
          Why: Missing colon after 'model' on line 5
          Fix: Add a colon after 'model:' and ensure proper indentation

    """

    def __init__(self, use_color: bool = True):
        """Initialize the error formatter.

        Args:
            use_color: Whether to use colored output for terminal display

        """
        self.use_color = use_color

    def format_error(
        self,
        error_type: ErrorType,
        message: str,
        what: str,
        why: str,
        fix: str,
        example: str | None = None,
        details: dict | None = None,
    ) -> str:
        """Format an error message with consistent structure.

        Args:
            error_type: Type of error (CONFIG, DATA, MODEL, etc.)
            message: Brief one-line description
            what: Detailed explanation of what happened
            why: Root cause or reason
            fix: How to resolve the issue
            example: Optional example command to demonstrate fix
            details: Optional dictionary of additional details

        Returns:
            Formatted error message string

        Examples:
            >>> formatter = ErrorFormatter(use_color=False)
            >>> error = formatter.format_error(
            ...     error_type=ErrorType.CONFIG,
            ...     message="Model type not found",
            ...     what="The model type 'xgboosted' is not recognized",
            ...     why="The model type may be misspelled",
            ...     fix="Check the model type spelling. Valid types: xgboost, lightgbm, logistic_regression",
            ...     example="model:\\n  type: xgboost"
            ... )
            >>> print(error)
            ERROR [CONFIG]: Model type not found
              What: The model type 'xgboosted' is not recognized
              Why: The model type may be misspelled
              Fix: Check the model type spelling. Valid types: xgboost, lightgbm, logistic_regression

        Example:
                model:
                  type: xgboost

        """
        lines = []

        # Header with error type and message
        header = f"ERROR [{error_type.value}]: {message}"
        lines.append(header)

        # What happened
        lines.append(f"  What: {what}")

        # Why it happened
        lines.append(f"  Why: {why}")

        # How to fix
        lines.append(f"  Fix: {fix}")

        # Add example if provided
        if example:
            lines.append("")
            lines.append("  Example:")
            # Indent example lines
            for line in example.split("\n"):
                lines.append(f"    {line}")

        # Add details if provided
        if details:
            lines.append("")
            lines.append("  Details:")
            for key, value in details.items():
                lines.append(f"    {key}: {value}")

        return "\n".join(lines)

    def print_error(self, formatted_error: str, exit_code: int | None = None) -> None:
        """Print a formatted error message to stderr.

        Args:
            formatted_error: The formatted error string
            exit_code: Optional exit code to use (will not exit if None)

        """
        if self.use_color:
            typer.secho(formatted_error, fg=typer.colors.RED, err=True)
        else:
            typer.echo(formatted_error, err=True)

        if exit_code is not None:
            raise typer.Exit(exit_code)

    # Convenience methods for common error types

    def format_config_error(
        self,
        message: str,
        what: str,
        why: str,
        fix: str,
        example: str | None = None,
    ) -> str:
        """Format a configuration error.

        Args:
            message: Brief description
            what: What happened
            why: Why it happened
            fix: How to fix
            example: Optional example

        Returns:
            Formatted error message

        """
        return self.format_error(
            error_type=ErrorType.CONFIG,
            message=message,
            what=what,
            why=why,
            fix=fix,
            example=example,
        )

    def format_data_error(
        self,
        message: str,
        what: str,
        why: str,
        fix: str,
        example: str | None = None,
    ) -> str:
        """Format a data-related error.

        Args:
            message: Brief description
            what: What happened
            why: Why it happened
            fix: How to fix
            example: Optional example

        Returns:
            Formatted error message

        """
        return self.format_error(
            error_type=ErrorType.DATA,
            message=message,
            what=what,
            why=why,
            fix=fix,
            example=example,
        )

    def format_model_error(
        self,
        message: str,
        what: str,
        why: str,
        fix: str,
        example: str | None = None,
    ) -> str:
        """Format a model-related error.

        Args:
            message: Brief description
            what: What happened
            why: Why it happened
            fix: How to fix
            example: Optional example

        Returns:
            Formatted error message

        """
        return self.format_error(
            error_type=ErrorType.MODEL,
            message=message,
            what=what,
            why=why,
            fix=fix,
            example=example,
        )

    def format_system_error(
        self,
        message: str,
        what: str,
        why: str,
        fix: str,
        example: str | None = None,
    ) -> str:
        """Format a system-level error.

        Args:
            message: Brief description
            what: What happened
            why: Why it happened
            fix: How to fix
            example: Optional example

        Returns:
            Formatted error message

        """
        return self.format_error(
            error_type=ErrorType.SYSTEM,
            message=message,
            what=what,
            why=why,
            fix=fix,
            example=example,
        )

    def format_validation_error(
        self,
        message: str,
        what: str,
        why: str,
        fix: str,
        example: str | None = None,
    ) -> str:
        """Format a validation error.

        Args:
            message: Brief description
            what: What happened
            why: Why it happened
            fix: How to fix
            example: Optional example

        Returns:
            Formatted error message

        """
        return self.format_error(
            error_type=ErrorType.VALIDATION,
            message=message,
            what=what,
            why=why,
            fix=fix,
            example=example,
        )

    def format_component_error(
        self,
        message: str,
        what: str,
        why: str,
        fix: str,
        example: str | None = None,
    ) -> str:
        """Format a component-related error.

        Args:
            message: Brief description
            what: What happened
            why: Why it happened
            fix: How to fix
            example: Optional example

        Returns:
            Formatted error message

        """
        return self.format_error(
            error_type=ErrorType.COMPONENT,
            message=message,
            what=what,
            why=why,
            fix=fix,
            example=example,
        )


# Global formatter instance for convenience
_default_formatter = ErrorFormatter()


def format_error(
    error_type: ErrorType,
    message: str,
    what: str,
    why: str,
    fix: str,
    example: str | None = None,
) -> str:
    """Convenience function to format an error using the default formatter.

    Args:
        error_type: Type of error
        message: Brief description
        what: What happened
        why: Why it happened
        fix: How to fix
        example: Optional example

    Returns:
        Formatted error message

    """
    return _default_formatter.format_error(
        error_type=error_type,
        message=message,
        what=what,
        why=why,
        fix=fix,
        example=example,
    )


def print_error(formatted_error: str, exit_code: int | None = None) -> None:
    """Convenience function to print an error using the default formatter.

    Args:
        formatted_error: The formatted error string
        exit_code: Optional exit code to use

    """
    _default_formatter.print_error(formatted_error, exit_code=exit_code)
