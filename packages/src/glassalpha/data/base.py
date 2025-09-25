"""Base data interface for different data modalities.

This module defines the common interface that all data loaders
must implement to enable consistent data handling across
different modalities (tabular, text, image).
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel


class DataSchema(BaseModel):
    """Base schema for data validation."""

    target: str
    features: list[str]
    sensitive_features: list[str] | None = None


class DataInterface(ABC):
    """Protocol for data loaders across different modalities."""

    @abstractmethod
    def load(self, path: Path, schema: DataSchema | None = None) -> pd.DataFrame:
        """Load data from file.

        Args:
            path: Path to data file
            schema: Optional schema for validation

        Returns:
            Loaded data as DataFrame

        """

    @abstractmethod
    def validate_schema(self, data: pd.DataFrame, schema: DataSchema) -> None:
        """Validate data against schema.

        Args:
            data: DataFrame to validate
            schema: Schema to validate against

        Raises:
            ValueError: If data doesn't match schema

        """

    @abstractmethod
    def extract_features_target(
        self,
        data: pd.DataFrame,
        schema: DataSchema,
    ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame | None]:
        """Extract features, target, and sensitive features.

        Args:
            data: Full dataset
            schema: Schema defining columns

        Returns:
            Tuple of (features, target, sensitive_features)

        """

    @abstractmethod
    def hash_data(self, data: pd.DataFrame) -> str:
        """Generate deterministic hash of data.

        Args:
            data: DataFrame to hash

        Returns:
            Hex string hash of data content

        """

    @abstractmethod
    def split_data(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify_column: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train/test sets.

        Args:
            data: Full dataset
            test_size: Fraction for test set
            random_state: Random seed for reproducibility
            stratify_column: Column to stratify split on

        Returns:
            Tuple of (train_data, test_data)

        """
