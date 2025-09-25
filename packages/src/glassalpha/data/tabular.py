"""Tabular data loading and preprocessing.

This module provides comprehensive tabular data handling with:
- CSV/Parquet loading with robust error handling
- Schema validation using Pydantic
- Protected attributes extraction for fairness analysis
- Deterministic dataset hashing for reproducibility
- Stratified train/test splitting with seed control
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.model_selection import train_test_split

from .base import DataInterface, DataSchema

logger = logging.getLogger(__name__)


class TabularDataSchema(DataSchema):
    """Schema for tabular data validation with additional constraints."""

    target: str = Field(..., description="Name of target column")
    features: list[str] = Field(..., min_items=1, description="List of feature column names")
    sensitive_features: list[str] | None = Field(None, description="Protected attributes for fairness analysis")
    categorical_features: list[str] | None = Field(None, description="Columns that should be treated as categorical")
    numeric_features: list[str] | None = Field(None, description="Columns that should be treated as numeric")

    def model_post_init(self, __context: Any) -> None:
        """Validate schema constraints after initialization."""
        # Target cannot be in features
        if self.target in self.features:
            raise ValueError(f"Target '{self.target}' cannot be in features list")

        # Sensitive features must be subset of features or standalone
        if self.sensitive_features:
            invalid_sensitive = set(self.sensitive_features) - set(self.features + [self.target])
            if invalid_sensitive:
                logger.warning(f"Sensitive features not in feature/target columns: {invalid_sensitive}")


class TabularDataLoader(DataInterface):
    """Comprehensive tabular data loader with validation and preprocessing."""

    def __init__(self):
        """Initialize tabular data loader."""
        self.supported_formats = {".csv", ".parquet", ".pkl", ".feather"}

    def load(self, path: Path, schema: DataSchema | None = None) -> pd.DataFrame:
        """Load tabular data from file.

        Args:
            path: Path to data file (CSV, Parquet, Pickle, Feather)
            schema: Optional schema for validation

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not supported or data invalid

        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        if path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {path.suffix}. Supported: {', '.join(self.supported_formats)}")

        logger.info(f"Loading data from {path}")

        try:
            # Load based on file format
            if path.suffix == ".csv":
                data = pd.read_csv(path)
            elif path.suffix == ".parquet":
                data = pd.read_parquet(path)
            elif path.suffix == ".pkl":
                data = pd.read_pickle(path)
            elif path.suffix == ".feather":
                data = pd.read_feather(path)
            else:
                raise ValueError(f"Format {path.suffix} not implemented")

        except Exception as e:
            raise ValueError(f"Failed to load data from {path}: {e}") from e

        logger.info(f"Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")

        # Validate schema if provided
        if schema:
            self.validate_schema(data, schema)

        return data

    def validate_schema(self, data: pd.DataFrame, schema: DataSchema) -> None:
        """Validate DataFrame against schema.

        Args:
            data: DataFrame to validate
            schema: Schema to validate against

        Raises:
            ValueError: If validation fails

        """
        logger.info("Validating data schema")

        # Check target column exists
        if schema.target not in data.columns:
            raise ValueError(f"Target column '{schema.target}' not found in data")

        # Check feature columns exist
        missing_features = set(schema.features) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        # Check sensitive features exist
        if schema.sensitive_features:
            missing_sensitive = set(schema.sensitive_features) - set(data.columns)
            if missing_sensitive:
                raise ValueError(f"Missing sensitive feature columns: {missing_sensitive}")

        # Additional validation for TabularDataSchema
        if isinstance(schema, TabularDataSchema):
            if schema.categorical_features:
                missing_cat = set(schema.categorical_features) - set(data.columns)
                if missing_cat:
                    logger.warning(f"Missing categorical columns: {missing_cat}")

            if schema.numeric_features:
                missing_num = set(schema.numeric_features) - set(data.columns)
                if missing_num:
                    logger.warning(f"Missing numeric columns: {missing_num}")

        # Check for missing values in critical columns
        critical_columns = [schema.target] + schema.features
        if schema.sensitive_features:
            critical_columns.extend(schema.sensitive_features)

        missing_counts = data[critical_columns].isnull().sum()
        if missing_counts.any():
            logger.warning(f"Missing values detected: {missing_counts[missing_counts > 0].to_dict()}")

        logger.info("Schema validation completed successfully")

    def extract_features_target(
        self,
        data: pd.DataFrame,
        schema: DataSchema,
    ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame | None]:
        """Extract features, target, and sensitive features from data.

        Args:
            data: Full dataset
            schema: Schema defining columns

        Returns:
            Tuple of (features_df, target_array, sensitive_df)

        """
        # Extract features
        X = data[schema.features].copy()

        # Extract target
        y = data[schema.target].values

        # Extract sensitive features if specified
        sensitive = None
        if schema.sensitive_features:
            sensitive = data[schema.sensitive_features].copy()

        logger.info(f"Extracted features: {X.shape}, target: {y.shape}")
        if sensitive is not None:
            logger.info(f"Extracted sensitive features: {sensitive.shape}")

        return X, y, sensitive

    def hash_data(self, data: pd.DataFrame) -> str:
        """Generate deterministic hash of DataFrame content.

        Args:
            data: DataFrame to hash

        Returns:
            SHA256 hex hash of data content

        """
        # Create deterministic string representation
        # Sort by columns first, then by index to ensure consistency
        data_sorted = data.sort_index(axis=1).sort_index(axis=0)

        # Convert to string with consistent formatting
        data_str = data_sorted.to_csv(index=False, float_format="%.10g")

        # Generate hash
        hash_obj = hashlib.sha256(data_str.encode("utf-8"))
        data_hash = hash_obj.hexdigest()

        logger.info(f"Generated data hash: {data_hash[:12]}...")
        return data_hash

    def split_data(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify_column: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train/test sets with stratification.

        Args:
            data: Full dataset
            test_size: Fraction for test set (0.0 to 1.0)
            random_state: Random seed for reproducibility
            stratify_column: Column to stratify split on (typically target)

        Returns:
            Tuple of (train_data, test_data)

        Raises:
            ValueError: If stratify_column not found or invalid parameters

        """
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")

        stratify = None
        if stratify_column:
            if stratify_column not in data.columns:
                raise ValueError(f"Stratify column '{stratify_column}' not found in data")
            stratify = data[stratify_column]

        logger.info(f"Splitting data: {len(data)} total, test_size={test_size}")

        try:
            train_data, test_data = train_test_split(
                data,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError as e:
            if stratify is not None:
                logger.warning(f"Stratified split failed: {e}. Trying without stratification.")
                train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
            else:
                raise

        logger.info(f"Split completed: train={len(train_data)}, test={len(test_data)}")

        return train_data, test_data

    def preprocess_features(
        self,
        X: pd.DataFrame,
        schema: TabularDataSchema,
        fit_preprocessor: bool = True,
    ) -> pd.DataFrame:
        """Basic preprocessing for tabular features.

        Args:
            X: Feature DataFrame
            schema: Schema with feature type information
            fit_preprocessor: Whether to fit preprocessing transformations

        Returns:
            Preprocessed feature DataFrame

        """
        X_processed = X.copy()

        # Handle categorical features
        if schema.categorical_features:
            cat_features = [f for f in schema.categorical_features if f in X_processed.columns]
            for col in cat_features:
                # Simple label encoding for now
                if X_processed[col].dtype == "object":
                    X_processed[col] = pd.Categorical(X_processed[col]).codes

        # Handle numeric features
        if schema.numeric_features:
            num_features = [f for f in schema.numeric_features if f in X_processed.columns]
            for col in num_features:
                # Ensure numeric type
                X_processed[col] = pd.to_numeric(X_processed[col], errors="coerce")

        # Fill missing values with appropriate defaults
        for col in X_processed.columns:
            if X_processed[col].dtype in ["int64", "float64"]:
                X_processed[col] = X_processed[col].fillna(X_processed[col].median())
            else:
                X_processed[col] = X_processed[col].fillna(
                    X_processed[col].mode().iloc[0] if not X_processed[col].mode().empty else 0,
                )

        logger.info(f"Preprocessing completed for {X_processed.shape[1]} features")

        return X_processed

    def get_data_summary(self, data: pd.DataFrame) -> dict[str, Any]:
        """Generate comprehensive data summary statistics.

        Args:
            data: DataFrame to summarize

        Returns:
            Dictionary with summary statistics

        """
        summary = {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024**2,
        }

        # Add numeric column statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary["numeric_stats"] = data[numeric_cols].describe().to_dict()

        # Add categorical column statistics
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            summary["categorical_stats"] = {}
            for col in categorical_cols:
                summary["categorical_stats"][col] = {
                    "unique_values": data[col].nunique(),
                    "most_frequent": data[col].mode().iloc[0] if not data[col].mode().empty else None,
                    "value_counts": data[col].value_counts().head().to_dict(),
                }

        return summary


# Convenience function for quick loading
def load_tabular_data(path: Path, schema: TabularDataSchema | None = None) -> pd.DataFrame:
    """Convenience function to load tabular data.

    Args:
        path: Path to data file
        schema: Optional schema for validation

    Returns:
        Loaded DataFrame

    """
    loader = TabularDataLoader()
    return loader.load(path, schema)


def create_schema_from_data(
    data: pd.DataFrame,
    target_column: str,
    sensitive_features: list[str] | None = None,
    categorical_threshold: int = 10,
) -> TabularDataSchema:
    """Create schema automatically from DataFrame.

    Args:
        data: DataFrame to analyze
        target_column: Name of target column
        sensitive_features: List of sensitive feature names
        categorical_threshold: Max unique values to consider column categorical

    Returns:
        Generated TabularDataSchema

    """
    # Feature columns (all except target)
    features = [col for col in data.columns if col != target_column]

    # Auto-detect categorical features
    categorical_features = []
    numeric_features = []

    for col in features:
        if data[col].dtype in ["object", "category"] or (
            data[col].nunique() <= categorical_threshold and data[col].dtype in ["int64", "bool"]
        ):
            categorical_features.append(col)
        else:
            numeric_features.append(col)

    return TabularDataSchema(
        target=target_column,
        features=features,
        sensitive_features=sensitive_features,
        categorical_features=categorical_features if categorical_features else None,
        numeric_features=numeric_features if numeric_features else None,
    )
