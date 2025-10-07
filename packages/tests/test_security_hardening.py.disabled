"""Test security hardening functionality."""

import logging
import random
import stat
import string
from unittest.mock import patch

import pytest
import yaml

from glassalpha.security import (
    sanitize_log_message,
    setup_secure_logging,
    validate_local_model,
    validate_model_uri,
)
from glassalpha.security.logs import LogSanitizer, log_security_event
from glassalpha.security.paths import SecurityError, get_default_allowed_dirs
from glassalpha.security.yaml_loader import YAMLSecurityError, safe_load_yaml


class TestModelPathValidation:
    """Test model path validation security."""

    def test_validate_local_model_basic(self, tmp_path):
        """Test basic local model validation."""
        # Create a test model file
        model_file = tmp_path / "model.json"
        model_file.write_text('{"model": "test"}')

        # Should validate successfully
        validated_path = validate_local_model(
            str(model_file),
            allowed_dirs=[str(tmp_path)],
        )

        assert validated_path == model_file.resolve()

    def test_validate_local_model_file_not_found(self):
        """Test validation fails for non-existent files."""
        with pytest.raises(SecurityError, match="Model file not found"):
            validate_local_model("/nonexistent/file.json")

    def test_validate_local_model_outside_allowed_dirs(self, tmp_path):
        """Test validation fails for files outside allowed directories."""
        model_file = tmp_path / "model.json"
        model_file.write_text('{"model": "test"}')

        # Try to validate with different allowed directory
        other_dir = tmp_path / "other"
        other_dir.mkdir()

        with pytest.raises(SecurityError, match="outside allowed directories"):
            validate_local_model(
                str(model_file),
                allowed_dirs=[str(other_dir)],
            )

    def test_validate_local_model_symlink_denied(self, tmp_path):
        """Test validation fails for symlinks by default."""
        # Create model file and symlink
        model_file = tmp_path / "model.json"
        model_file.write_text('{"model": "test"}')

        symlink_file = tmp_path / "model_link.json"
        symlink_file.symlink_to(model_file)

        # Should fail by default
        with pytest.raises(SecurityError, match="Symbolic links are not allowed"):
            validate_local_model(
                str(symlink_file),
                allowed_dirs=[str(tmp_path)],
            )

        # Should succeed when explicitly allowed
        validated_path = validate_local_model(
            str(symlink_file),
            allowed_dirs=[str(tmp_path)],
            allow_symlinks=True,
        )
        assert validated_path == symlink_file.resolve()

    def test_validate_local_model_world_writable_denied(self, tmp_path):
        """Test validation fails for world-writable files by default."""
        model_file = tmp_path / "model.json"
        model_file.write_text('{"model": "test"}')

        # Make file world-writable
        model_file.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)

        # Should fail by default
        with pytest.raises(SecurityError, match="World-writable files are not allowed"):
            validate_local_model(
                str(model_file),
                allowed_dirs=[str(tmp_path)],
            )

        # Should succeed when explicitly allowed
        validated_path = validate_local_model(
            str(model_file),
            allowed_dirs=[str(tmp_path)],
            allow_world_writable=True,
        )
        assert validated_path == model_file.resolve()

    def test_validate_local_model_size_limit(self, tmp_path):
        """Test validation fails for files exceeding size limit."""
        model_file = tmp_path / "large_model.json"

        # Create file larger than 1MB
        large_content = "x" * (2 * 1024 * 1024)  # 2MB
        model_file.write_text(large_content)

        # Should fail with 1MB limit
        with pytest.raises(SecurityError, match="Model file too large"):
            validate_local_model(
                str(model_file),
                allowed_dirs=[str(tmp_path)],
                max_size_mb=1.0,
            )

        # Should succeed with higher limit
        validated_path = validate_local_model(
            str(model_file),
            allowed_dirs=[str(tmp_path)],
            max_size_mb=5.0,
        )
        assert validated_path == model_file.resolve()

    def test_validate_local_model_hash_verification(self, tmp_path):
        """Test SHA-256 hash verification."""
        model_file = tmp_path / "model.json"
        content = '{"model": "test"}'
        model_file.write_text(content)

        # Calculate expected hash
        import hashlib

        expected_hash = hashlib.sha256(content.encode()).hexdigest()

        # Should succeed with correct hash
        validated_path = validate_local_model(
            str(model_file),
            allowed_dirs=[str(tmp_path)],
            expected_sha256=expected_hash,
        )
        assert validated_path == model_file.resolve()

        # Should fail with wrong hash
        wrong_hash = "0" * 64
        with pytest.raises(SecurityError, match="Model file hash mismatch"):
            validate_local_model(
                str(model_file),
                allowed_dirs=[str(tmp_path)],
                expected_sha256=wrong_hash,
            )

    def test_validate_model_uri_file_scheme(self, tmp_path):
        """Test file:// URI validation."""
        model_file = tmp_path / "model.json"
        model_file.write_text('{"model": "test"}')

        file_uri = f"file://{model_file}"

        # Need to include tmp_path in allowed directories for this test
        with patch("glassalpha.security.paths.validate_local_model") as mock_validate:
            mock_validate.return_value = model_file

            result = validate_model_uri(file_uri, allowed_schemes=["file"])

            assert result["scheme"] == "file"
            assert result["is_remote"] is False
            assert result["validated"] is True

    def test_validate_model_uri_remote_disabled(self):
        """Test remote URIs are disabled by default."""
        with pytest.raises(SecurityError, match="URI scheme 'https' not allowed"):
            validate_model_uri("https://example.com/model.json")

    def test_validate_model_uri_remote_enabled(self):
        """Test remote URIs work when explicitly enabled."""
        result = validate_model_uri(
            "https://example.com/model.json",
            allow_remote=True,
            allowed_schemes=["https"],
        )

        assert result["scheme"] == "https"
        assert result["is_remote"] is True
        assert result["hostname"] == "example.com"

    def test_validate_model_uri_disallowed_scheme(self):
        """Test validation fails for disallowed schemes."""
        with pytest.raises(SecurityError, match="URI scheme 'ftp' not allowed"):
            validate_model_uri("ftp://example.com/model.json")

    def test_get_default_allowed_dirs(self):
        """Test default allowed directories."""
        defaults = get_default_allowed_dirs()

        assert "." in defaults
        assert "~/models" in defaults
        assert "./models" in defaults
        assert isinstance(defaults, list)


class TestYAMLSecurity:
    """Test YAML loading security."""

    def test_safe_load_yaml_basic(self, tmp_path):
        """Test basic YAML loading."""
        yaml_file = tmp_path / "config.yaml"
        yaml_content = {
            "model": {"type": "xgboost"},
            "data": {"dataset": "custom", "path": "data.csv"},
        }

        with yaml_file.open("w") as f:
            yaml.dump(yaml_content, f)

        loaded = safe_load_yaml(yaml_file)
        assert loaded == yaml_content

    def test_safe_load_yaml_from_string(self):
        """Test YAML loading from string."""
        yaml_string = """
        model:
          type: xgboost
        data:
          path: data.csv
        """

        loaded = safe_load_yaml(yaml_string)
        assert loaded["model"]["type"] == "xgboost"
        assert loaded["data"]["path"] == "data.csv"

    def test_safe_load_yaml_file_size_limit(self, tmp_path):
        """Test YAML file size limits."""
        yaml_file = tmp_path / "large.yaml"

        # Create large YAML content
        large_content = "key: " + "x" * (5 * 1024 * 1024)  # 5MB
        yaml_file.write_text(large_content)

        # Should fail with 1MB limit
        with pytest.raises(YAMLSecurityError, match="YAML file too large"):
            safe_load_yaml(yaml_file, max_file_size_mb=1.0)

    def test_safe_load_yaml_depth_limit(self):
        """Test YAML nesting depth limits."""
        # Create deeply nested YAML
        nested_yaml = "root:"
        for i in range(25):  # Deeper than default limit of 20
            nested_yaml += f"\n{'  ' * (i + 1)}level{i}:"
        nested_yaml += f"\n{'  ' * 26}value: test"

        # Should fail with default depth limit
        with pytest.raises(YAMLSecurityError, match="YAML nesting too deep"):
            safe_load_yaml(nested_yaml)

        # Should succeed with higher limit
        loaded = safe_load_yaml(nested_yaml, max_depth=30)
        assert isinstance(loaded, dict)

    def test_safe_load_yaml_key_limit(self):
        """Test YAML key count limits."""
        # Create YAML with many keys
        many_keys = {f"key_{i}": f"value_{i}" for i in range(1500)}
        yaml_string = yaml.dump(many_keys)

        # Should fail with default key limit (1000)
        with pytest.raises(YAMLSecurityError, match="Too many keys in YAML"):
            safe_load_yaml(yaml_string)

        # Should succeed with higher limit
        loaded = safe_load_yaml(yaml_string, max_keys=2000)
        assert len(loaded) == 1500

    def test_safe_load_yaml_type_restrictions(self):
        """Test YAML type restrictions."""
        # This would normally be dangerous, but safe_load prevents it
        yaml_string = """
        model:
          type: xgboost
        data:
          path: data.csv
        """

        loaded = safe_load_yaml(yaml_string)

        # Should only contain safe types
        assert isinstance(loaded, dict)
        assert isinstance(loaded["model"], dict)
        assert isinstance(loaded["model"]["type"], str)

    def test_safe_load_yaml_invalid_yaml(self):
        """Test handling of invalid YAML."""
        invalid_yaml = """
        model:
          type: xgboost
        data:
          path: data.csv
        invalid: [unclosed list
        """

        with pytest.raises(YAMLSecurityError, match="YAML parsing failed"):
            safe_load_yaml(invalid_yaml)

    def test_safe_load_yaml_non_dict_root(self):
        """Test handling of non-dictionary root."""
        list_yaml = """
        - item1
        - item2
        """

        with pytest.raises(YAMLSecurityError, match="YAML root must be a dictionary"):
            safe_load_yaml(list_yaml)


class TestLogSanitization:
    """Test log sanitization security."""

    def test_log_sanitizer_api_keys(self):
        """Test API key sanitization."""
        sanitizer = LogSanitizer()

        message = "Using API key: abc123def456ghi789"
        sanitized = sanitizer.sanitize(message)

        assert "abc123def456ghi789" not in sanitized
        # Check for redaction marker without exact string match
        assert "[REDACTED" in sanitized and "]" in sanitized

    def test_log_sanitizer_passwords(self):
        """Test password sanitization."""
        sanitizer = LogSanitizer()

        messages = [
            'password="secret123"',
            "passwd: mypassword",
            "pwd=topsecret",
        ]

        for message in messages:
            sanitized = sanitizer.sanitize(message)
            assert "secret123" not in sanitized
            assert "mypassword" not in sanitized
            assert "topsecret" not in sanitized
            # Check for generic redaction marker
            assert "[REDACTED" in sanitized and "]" in sanitized

    def test_log_sanitizer_email_addresses(self):
        """Test email address sanitization."""
        sanitizer = LogSanitizer()

        message = "User email: john.doe@example.com"
        sanitized = sanitizer.sanitize(message)

        assert "john.doe@example.com" not in sanitized
        assert "john.doe***@example.com" in sanitized

    def test_log_sanitizer_ip_addresses(self):
        """Test IP address sanitization."""
        sanitizer = LogSanitizer()

        message = "Connecting to server 192.168.1.100"
        sanitized = sanitizer.sanitize(message)

        assert "192.168.1.100" not in sanitized
        assert "192.168.***.100" in sanitized

    def test_log_sanitizer_file_paths(self):
        """Test file path sanitization."""
        sanitizer = LogSanitizer()

        messages = [
            "Loading from /Users/john/model.json",
            "Saving to /home/jane/output.pdf",
            "Path: C:\\Users\\Bob\\data.csv",
        ]

        for message in messages:
            sanitized = sanitizer.sanitize(message)
            assert "john" not in sanitized
            assert "jane" not in sanitized
            assert "Bob" not in sanitized
            # Check for generic user replacement
            assert "[USER]" in sanitized or "home/[USER]" in sanitized or "C:\\Users\\[USER]" in sanitized

    def test_log_sanitizer_credit_cards(self):
        """Test credit card number sanitization."""
        sanitizer = LogSanitizer()

        message = "Payment with card 4532-1234-5678-9012"
        sanitized = sanitizer.sanitize(message)

        assert "4532-1234-5678-9012" not in sanitized
        # Check for generic redaction marker
        assert "[REDACTED" in sanitized and "CC]" in sanitized

    def test_log_sanitizer_control_characters(self):
        """Test control character sanitization."""
        sanitizer = LogSanitizer()

        message = "Data: \x00\x01\x02test\x7f\x80"
        sanitized = sanitizer.sanitize(message)

        assert "\x00" not in sanitized
        assert "\x01" not in sanitized
        assert "[CTRL]" in sanitized

    def test_log_sanitizer_custom_pattern(self):
        """Test custom sanitization patterns."""
        sanitizer = LogSanitizer()

        # Add custom pattern
        sanitizer.add_pattern(r"SECRET_\w+", "[CUSTOM_SECRET]")

        message = "Using SECRET_KEY123 for authentication"
        sanitized = sanitizer.sanitize(message)

        assert "SECRET_KEY123" not in sanitized
        assert "[CUSTOM_SECRET]" in sanitized

    @pytest.mark.parametrize("sanitizer_func", ["standalone", "class_instance"])
    def test_sanitizer_combined_functionality(self, sanitizer_func):
        """Test sanitization using both standalone and class methods."""
        message = "API key: abc123 and password: secret"

        if sanitizer_func == "standalone":
            sanitized = sanitize_log_message(message)
        else:
            sanitizer = LogSanitizer()
            sanitized = sanitizer.sanitize(message)

        assert "abc123" not in sanitized
        assert "secret" not in sanitized
        # Check for generic redaction markers
        assert "[REDACTED" in sanitized and "]" in sanitized

    @pytest.mark.parametrize("iteration", range(10))  # Run 10 times for fuzzing
    def test_sanitizer_fuzzing_random_secrets(self, iteration):
        """Fuzz test with random secret-like strings."""
        sanitizer = LogSanitizer()

        # Generate random secret-like strings that match sanitization patterns
        secret_types = ["password", "api_key", "secret"]  # Focus on types with patterns
        secret_chars = string.ascii_letters + string.digits  # Only alphanumeric to match patterns

        random_secret = "".join(random.choices(secret_chars, k=random.randint(8, 20)))
        secret_type = random.choice(secret_types)
        message = f"{secret_type}: {random_secret}"

        sanitized = sanitizer.sanitize(message)

        # Assert no unredacted secrets remain
        assert random_secret not in sanitized
        # Ensure some form of redaction occurred
        assert "[REDACTED" in sanitized and "]" in sanitized

    def test_setup_secure_logging(self):
        """Test secure logging setup."""
        logger = setup_secure_logging("test_logger", level=logging.INFO)

        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

        # Test that messages are sanitized
        with patch("logging.StreamHandler.emit") as mock_emit:
            logger.info("API key: %s", "secret123")

            # Check that emit was called
            assert mock_emit.called

            # The actual sanitization happens in the formatter
            # so we can't easily test the final output here

    def test_log_security_event(self):
        """Test security event logging."""
        with patch("logging.Logger.log") as mock_log:
            log_security_event(
                "unauthorized_access",
                {"user": "test@example.com", "ip": "192.168.1.1"},
                "WARNING",
            )

            assert mock_log.called
            args, kwargs = mock_log.call_args

            # Check that the event was logged with correct level
            assert args[0] == logging.WARNING
            assert "Security Event" in args[1]

            # Check that details were included
            assert "event_type" in kwargs["extra"]
            assert kwargs["extra"]["event_type"] == "unauthorized_access"


class TestSecurityIntegration:
    """Test security integration with main system."""

    def test_security_config_schema(self):
        """Test security configuration schema."""
        from glassalpha.config.schema import SecurityConfig

        # Test default configuration
        config = SecurityConfig()

        assert config.strict is False
        assert isinstance(config.model_paths, dict)
        assert isinstance(config.yaml_loading, dict)
        assert isinstance(config.logging, dict)

        # Test custom configuration
        custom_config = SecurityConfig(
            strict=True,
            model_paths={"allowed_dirs": ["/secure/models"]},
        )

        assert custom_config.strict is True
        assert custom_config.model_paths["allowed_dirs"] == ["/secure/models"]

    def test_audit_config_with_security(self):
        """Test AuditConfig includes security configuration."""
        from glassalpha.config.schema import AuditConfig

        config_dict = {
            "audit_profile": "test",
            "model": {"type": "xgboost"},
            "data": {"dataset": "custom", "path": "data.csv"},
            "security": {
                "strict": True,
                "model_paths": {"allowed_dirs": ["/secure"]},
            },
        }

        config = AuditConfig(**config_dict)

        assert config.security.strict is True
        assert config.security.model_paths["allowed_dirs"] == ["/secure"]

    def test_security_config_defaults(self):
        """Test security configuration has sensible defaults."""
        from glassalpha.config.schema import SecurityConfig

        config = SecurityConfig()

        # Model path defaults
        assert "." in config.model_paths["allowed_dirs"]
        assert config.model_paths["allow_remote"] is False
        assert config.model_paths["max_size_mb"] == 256.0

        # YAML loading defaults
        assert config.yaml_loading["max_file_size_mb"] == 10.0
        assert config.yaml_loading["max_depth"] == 20

        # Logging defaults
        assert config.logging["sanitize_messages"] is True
        assert config.logging["enable_json"] is False


if __name__ == "__main__":
    pytest.main([__file__])
