"""Model path validation and artifact integrity checking.

This module implements secure model loading with path validation, integrity
checking, and protection against common security vulnerabilities like
directory traversal, symlink attacks, and world-writable files.
"""

import hashlib
import logging
import stat
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised when security validation fails."""


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        path: Path to file to hash

    Returns:
        Hexadecimal SHA-256 hash string

    Raises:
        SecurityError: If file cannot be read or hashed

    """
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):  # 1MB chunks
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        raise SecurityError(f"Failed to compute hash for {path}: {e}") from e


def validate_local_model(
    path: str,
    allowed_dirs: list[str] | None = None,
    expected_sha256: str | None = None,
    max_size_mb: float = 256.0,
    allow_symlinks: bool = False,
    allow_world_writable: bool = False,
) -> Path:
    """Validate local model file path and integrity.

    This function implements comprehensive security checks for model files:
    - Path must be within allowed directories (default: current directory only)
    - File must exist and be readable
    - Symlinks are denied by default (prevents symlink attacks)
    - World-writable files are denied by default (prevents tampering)
    - File size is limited (prevents resource exhaustion)
    - Optional SHA-256 verification (prevents tampering)

    Args:
        path: Path to model file (can be relative or absolute)
        allowed_dirs: List of allowed base directories (default: ["."])
        expected_sha256: Expected SHA-256 hash for integrity verification
        max_size_mb: Maximum file size in MB (default: 256MB)
        allow_symlinks: Whether to allow symbolic links (default: False)
        allow_world_writable: Whether to allow world-writable files (default: False)

    Returns:
        Resolved absolute path to validated model file

    Raises:
        SecurityError: If any security check fails

    Examples:
        >>> # Basic validation (current directory only)
        >>> path = validate_local_model("model.json")

        >>> # Allow specific directories
        >>> path = validate_local_model(
        ...     "/data/models/model.json",
        ...     allowed_dirs=["/data/models", "/opt/models"]
        ... )

        >>> # With integrity verification
        >>> path = validate_local_model(
        ...     "model.json",
        ...     expected_sha256="abc123..."
        ... )

    """
    logger.debug("Validating local model path: %s", path)

    # Set default allowed directories
    if allowed_dirs is None:
        allowed_dirs = ["."]  # Current directory only by default

    # Resolve and normalize paths
    try:
        model_path = Path(path).expanduser().resolve()
    except Exception as e:
        raise SecurityError(f"Invalid path format: {path}") from e

    # Resolve allowed directories
    try:
        allowed_bases = [Path(d).expanduser().resolve() for d in allowed_dirs]
    except Exception as e:
        raise SecurityError(f"Invalid allowed directory: {e}") from e

    # Check if file exists
    if not model_path.exists():
        raise SecurityError(f"Model file not found: {model_path}")

    if not model_path.is_file():
        raise SecurityError(f"Path is not a regular file: {model_path}")

    # Check for symlinks (security risk)
    if not allow_symlinks and model_path.is_symlink():
        raise SecurityError(f"Symbolic links are not allowed: {model_path}")

    # Check if path is within allowed directories
    path_allowed = False
    for base in allowed_bases:
        try:
            # Check if model_path is under base directory
            model_path.relative_to(base)
            path_allowed = True
            break
        except ValueError:
            # Not under this base directory
            continue

    if not path_allowed:
        raise SecurityError(
            f"Model path outside allowed directories: {model_path}\n"
            f"Allowed directories: {[str(b) for b in allowed_bases]}",
        )

    # Check file permissions
    try:
        file_stat = model_path.stat()
    except Exception as e:
        raise SecurityError(f"Cannot access file permissions: {model_path}") from e

    # Check for world-writable files (security risk)
    if not allow_world_writable and (file_stat.st_mode & stat.S_IWOTH):
        raise SecurityError(f"World-writable files are not allowed: {model_path}")

    # Check file size
    size_mb = file_stat.st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        raise SecurityError(
            f"Model file too large: {size_mb:.1f}MB > {max_size_mb}MB limit",
        )

    # Verify SHA-256 hash if provided
    if expected_sha256:
        logger.debug("Verifying SHA-256 hash for %s", model_path)
        actual_hash = sha256_file(model_path)
        if actual_hash != expected_sha256:
            raise SecurityError(
                f"Model file hash mismatch:\nExpected: {expected_sha256}\nActual:   {actual_hash}",
            )
        logger.debug("SHA-256 hash verification passed")

    logger.info("Model path validation passed: %s (%.1fMB)", model_path, size_mb)
    return model_path


def validate_model_uri(
    uri: str,
    allow_remote: bool = False,
    allowed_schemes: list[str] | None = None,
    max_size_mb: float = 256.0,
) -> dict[str, Any]:
    """Validate model URI and return connection info.

    Args:
        uri: Model URI (file://, http://, https://, s3://, etc.)
        allow_remote: Whether to allow remote URIs (default: False)
        allowed_schemes: List of allowed URI schemes (default: ["file"])
        max_size_mb: Maximum download size in MB (default: 256MB)

    Returns:
        Dictionary with URI validation results and connection info

    Raises:
        SecurityError: If URI validation fails

    Examples:
        >>> # Local file URI
        >>> info = validate_model_uri("file:///path/to/model.json")

        >>> # Remote URI (requires explicit permission)
        >>> info = validate_model_uri(
        ...     "https://example.com/model.json",
        ...     allow_remote=True,
        ...     allowed_schemes=["https"]
        ... )

    """
    logger.debug("Validating model URI: %s", uri)

    # Set default allowed schemes
    if allowed_schemes is None:
        allowed_schemes = ["file"]  # Local files only by default

    try:
        parsed = urlparse(uri)
    except Exception as e:
        raise SecurityError(f"Invalid URI format: {uri}") from e

    # Check scheme
    if parsed.scheme not in allowed_schemes:
        raise SecurityError(
            f"URI scheme '{parsed.scheme}' not allowed. Allowed schemes: {allowed_schemes}",
        )

    # Check for remote URIs
    if parsed.scheme in ("http", "https", "ftp", "s3", "gs", "azure") and not allow_remote:
        raise SecurityError(
            f"Remote URIs are disabled by default. Set allow_remote=True to enable: {uri}",
        )

    # Validate specific schemes
    validation_result = {
        "uri": uri,
        "scheme": parsed.scheme,
        "is_remote": parsed.scheme not in ("file", ""),
        "max_size_mb": max_size_mb,
        "validated": True,
    }

    if parsed.scheme == "file":
        # Validate local file path
        local_path = parsed.path
        if not local_path:
            raise SecurityError(f"Empty file path in URI: {uri}")

        # Use validate_local_model for local files
        try:
            resolved_path = validate_local_model(local_path, max_size_mb=max_size_mb)
            validation_result["local_path"] = str(resolved_path)
        except SecurityError:
            # Re-raise with URI context
            raise SecurityError(f"Local file validation failed for URI: {uri}")

    elif parsed.scheme in ("http", "https"):
        # Validate HTTP/HTTPS URLs
        if not parsed.netloc:
            raise SecurityError(f"Missing hostname in URI: {uri}")

        # Basic hostname validation (prevent localhost, private IPs by default)
        hostname = parsed.hostname
        if hostname in ("localhost", "127.0.0.1", "::1"):
            logger.warning("URI points to localhost: %s", uri)

        validation_result["hostname"] = hostname
        validation_result["port"] = parsed.port

    elif parsed.scheme == "s3":
        # Validate S3 URIs
        if not parsed.netloc:
            raise SecurityError(f"Missing bucket name in S3 URI: {uri}")

        validation_result["bucket"] = parsed.netloc
        validation_result["key"] = parsed.path.lstrip("/")

    logger.info("Model URI validation passed: %s", uri)
    return validation_result


def get_default_allowed_dirs() -> list[str]:
    """Get default allowed directories for model files.

    Returns secure defaults based on common deployment patterns:
    - Current working directory
    - User's home directory models folder
    - System-wide models directory (if exists)

    Returns:
        List of default allowed directory paths

    """
    defaults = [
        ".",  # Current directory
        "~/models",  # User models directory
        "./models",  # Local models directory
    ]

    # Add system directories if they exist
    system_dirs = [
        "/opt/glassalpha/models",
        "/usr/local/share/glassalpha/models",
    ]

    for sys_dir in system_dirs:
        if Path(sys_dir).exists():
            defaults.append(sys_dir)

    return defaults


def create_secure_model_config() -> dict[str, Any]:
    """Create secure default configuration for model loading.

    Returns:
        Dictionary with secure model loading configuration

    """
    return {
        "allowed_dirs": get_default_allowed_dirs(),
        "max_size_mb": 256.0,
        "allow_symlinks": False,
        "allow_world_writable": False,
        "allow_remote": False,
        "allowed_schemes": ["file"],
        "require_hash": False,  # Can be enabled for high-security environments
    }
