"""Test schema field renaming (no more field shadowing)."""

import pytest

from glassalpha.config.schema import DataConfig


def test_data_schema_field_works():
    """Test that the data_schema field works correctly."""
    schema_data = {"type": "object", "properties": {"name": {"type": "string"}}}

    config = DataConfig(data_schema=schema_data)

    assert config.data_schema == schema_data


def test_no_schema_field_exists():
    """Test that the old 'schema' field no longer exists (no field shadowing)."""
    # The old 'schema' field should not exist anymore
    assert "schema" not in DataConfig.model_fields

    # Only data_schema should exist
    assert "data_schema" in DataConfig.model_fields


def test_data_schema_field_none_by_default():
    """Test that data_schema is None by default."""
    config = DataConfig()

    assert config.data_schema is None


def test_data_schema_field_not_deprecated():
    """Test that the data_schema field is not marked as deprecated."""
    # Get the field info for the data_schema field
    data_schema_field = DataConfig.model_fields["data_schema"]

    # Should not be marked as deprecated
    assert data_schema_field.deprecated is False or data_schema_field.deprecated is None
    assert "[DEPRECATED]" not in data_schema_field.description


def test_no_field_shadowing_warning():
    """Test that there's no field shadowing warning anymore."""
    import warnings  # noqa: PLC0415

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Creating a DataConfig should not trigger field shadowing warnings
        config = DataConfig(data_schema={"type": "object"})

        # Filter out any warnings that aren't about field shadowing
        shadowing_warnings = [warning for warning in w if "shadows" in str(warning.message).lower()]

        # Should have no field shadowing warnings
        assert len(shadowing_warnings) == 0


def test_complex_schema_structures():
    """Test that complex schema structures work correctly."""
    complex_schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0},
                    "emails": {
                        "type": "array",
                        "items": {"type": "string", "format": "email"},
                    },
                },
                "required": ["name"],
            },
        },
        "required": ["user"],
    }

    config = DataConfig(data_schema=complex_schema)

    # Should store correctly
    assert config.data_schema == complex_schema


def test_other_fields_preserved():
    """Test that other fields work correctly alongside data_schema."""
    schema_data = {"type": "object"}

    config = DataConfig(
        data_schema=schema_data,
        protected_attributes=["age", "gender"],
        target_column="outcome",
        feature_columns=["feature1", "feature2"],
    )

    # All fields should be preserved
    assert config.data_schema == schema_data
    assert config.protected_attributes == ["age", "gender"]
    assert config.target_column == "outcome"
    assert config.feature_columns == ["feature1", "feature2"]


def test_model_dump_includes_data_schema():
    """Test that model_dump includes data_schema field."""
    schema_data = {"type": "object"}

    config = DataConfig(data_schema=schema_data)
    dump = config.model_dump()

    assert "data_schema" in dump
    assert dump["data_schema"] == schema_data
    # Old schema field should not exist
    assert "schema" not in dump


def test_json_schema_structure():
    """Test that JSON schema has correct structure."""
    json_schema = DataConfig.model_json_schema()

    # Should have data_schema field
    assert "data_schema" in json_schema["properties"]

    # Should not have old schema field
    assert "schema" not in json_schema["properties"]

    # data_schema field should not be deprecated
    data_schema_prop = json_schema["properties"]["data_schema"]
    assert data_schema_prop.get("deprecated") is not True


def test_strict_mode_validation_works():
    """Test that strict mode validation works with data_schema field."""
    from glassalpha.config.schema import AuditConfig
    from glassalpha.config.strict import validate_strict_mode

    # Create minimal config that should pass basic data schema check
    config_dict = {
        "audit_profile": "test",
        "data": {
            "path": "/path/to/data.csv",
            "data_schema": {"type": "object"},  # Using new field
            "protected_attributes": ["age"],
            "target_column": "outcome",
        },
        "model": {
            "type": "xgboost",
            "path": "/path/to/model.json",  # Add required model path
        },
        "explainers": {
            "priority": ["treeshap"],  # Add required explainer priority
        },
        "strict_mode": True,
    }

    config = AuditConfig(**config_dict)

    # Should not raise validation errors for data schema part
    # (Other parts may fail, but data schema validation should pass)
    try:
        validate_strict_mode(config)
    except Exception as e:
        # Check that the error is NOT about missing data schema
        error_msg = str(e)
        assert "Data schema must be specified" not in error_msg, f"Data schema validation failed: {e}"


if __name__ == "__main__":
    pytest.main([__file__])
