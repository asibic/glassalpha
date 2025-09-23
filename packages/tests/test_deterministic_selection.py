"""Tests for deterministic plugin selection.

These tests ensure that component selection is reproducible and
deterministic, which is critical for regulatory compliance.
"""

import hashlib
import json

# Mock pandas and numpy for testing
import sys
from unittest.mock import MagicMock

sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()

from glassalpha.core import (
    ExplainerRegistry,
    MetricRegistry,
    ModelRegistry,
    list_components,
    select_explainer,
)


class TestDeterministicSelection:
    """Test suite for deterministic component selection."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear registries to ensure clean state
        # Note: In production, registries persist, but for tests we want isolation
        pass

    def test_explainer_selection_is_deterministic(self):
        """Test that explainer selection is deterministic with same config."""
        config = {
            "explainers": {
                "priority": ["noop", "nonexistent", "another_fake"]
            }
        }

        # Run selection multiple times
        results = []
        for _ in range(10):
            selected = select_explainer("xgboost", config)
            results.append(selected)

        # All selections should be identical
        assert len(set(results)) == 1, "Selection is not deterministic"
        assert results[0] == "noop", "Should select first available explainer"

    def test_priority_order_respected(self):
        """Test that priority order is respected in selection."""
        # Register additional test explainers
        @ExplainerRegistry.register("test_high_priority", priority=100)
        class TestHighPriority:
            capabilities = {"supported_models": ["xgboost"]}
            version = "1.0.0"
            priority = 100

        @ExplainerRegistry.register("test_low_priority", priority=1)
        class TestLowPriority:
            capabilities = {"supported_models": ["xgboost"]}
            version = "1.0.0"
            priority = 1

        # Config with specific priority
        config = {
            "explainers": {
                "priority": ["test_low_priority", "test_high_priority", "noop"]
            }
        }

        # Should select based on config priority, not internal priority
        selected = select_explainer("xgboost", config)
        assert selected == "test_low_priority", "Config priority should override internal priority"

    def test_fallback_to_noop(self):
        """Test fallback to NoOp explainer when others unavailable."""
        config = {
            "explainers": {
                "priority": ["nonexistent1", "nonexistent2", "noop"]
            }
        }

        selected = select_explainer("unknown_model", config)
        assert selected == "noop", "Should fallback to noop explainer"

    def test_selection_with_model_compatibility(self):
        """Test that model compatibility affects selection."""
        # Register model-specific explainer
        @ExplainerRegistry.register("test_specific", priority=50)
        class TestSpecific:
            capabilities = {"supported_models": ["specific_model"]}
            version = "1.0.0"
            priority = 50

        config = {
            "explainers": {
                "priority": ["test_specific", "noop"]
            }
        }

        # Should skip incompatible explainer
        selected = select_explainer("different_model", config)
        assert selected == "noop", "Should skip incompatible explainer"

        # Should select compatible explainer
        selected = select_explainer("specific_model", config)
        assert selected == "test_specific", "Should select compatible explainer"

    def test_empty_priority_list(self):
        """Test behavior with empty priority list."""
        config = {
            "explainers": {
                "priority": []
            }
        }

        # Should fall back to all registered explainers by priority
        selected = select_explainer("xgboost", config)
        assert selected is not None, "Should select from all available explainers"

    def test_selection_hash_consistency(self):
        """Test that same config produces same selection hash."""
        config1 = {
            "explainers": {
                "priority": ["noop", "test1", "test2"]
            }
        }

        config2 = {
            "explainers": {
                "priority": ["noop", "test1", "test2"]  # Same
            }
        }

        config3 = {
            "explainers": {
                "priority": ["test1", "noop", "test2"]  # Different order
            }
        }

        # Hash the config to ensure determinism
        def config_hash(cfg):
            return hashlib.sha256(
                json.dumps(cfg, sort_keys=True).encode()
            ).hexdigest()

        hash1 = config_hash(config1)
        hash2 = config_hash(config2)
        hash3 = config_hash(config3)

        assert hash1 == hash2, "Same config should produce same hash"
        assert hash1 != hash3, "Different config should produce different hash"

        # Same hash should lead to same selection
        selected1 = select_explainer("xgboost", config1)
        selected2 = select_explainer("xgboost", config2)
        select_explainer("xgboost", config3)

        assert selected1 == selected2, "Same config hash should give same selection"
        # Note: selected3 might be different due to different priority order

    def test_registry_list_is_deterministic(self):
        """Test that component listing is deterministic."""
        # Get components multiple times
        results = []
        for _ in range(5):
            components = list_components()
            # Convert to sorted JSON for comparison
            results.append(json.dumps(components, sort_keys=True))

        # All listings should be identical
        assert len(set(results)) == 1, "Component listing is not deterministic"

    def test_selection_with_missing_registry_entry(self):
        """Test graceful handling of missing registry entries."""
        config = {
            "explainers": {
                "priority": ["definitely_not_registered", "also_not_there", "noop"]
            }
        }

        # Should gracefully skip missing and select noop
        selected = select_explainer("xgboost", config)
        assert selected == "noop", "Should handle missing registry entries gracefully"

    def test_concurrent_selection_determinism(self):
        """Test that concurrent selections are still deterministic."""
        import threading

        config = {
            "explainers": {
                "priority": ["noop"]
            }
        }

        results = []
        lock = threading.Lock()

        def select_and_store():
            selected = select_explainer("xgboost", config)
            with lock:
                results.append(selected)

        # Create multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=select_and_store)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # All selections should be identical
        assert len(set(results)) == 1, "Concurrent selection is not deterministic"
        assert results[0] == "noop", "All threads should select noop"


class TestMetricSelection:
    """Test deterministic metric selection."""

    def test_metric_registry_determinism(self):
        """Test that metric selection is deterministic."""
        # Register test metrics
        @MetricRegistry.register("test_metric1", priority=10)
        class TestMetric1:
            metric_type = "test"
            version = "1.0.0"

        @MetricRegistry.register("test_metric2", priority=20)
        class TestMetric2:
            metric_type = "test"
            version = "1.0.0"

        # Get all metrics multiple times
        results = []
        for _ in range(5):
            metrics = MetricRegistry.get_all()
            # Get keys in consistent order
            metric_names = sorted(metrics.keys())
            results.append(metric_names)

        # All results should be identical
        assert all(r == results[0] for r in results), "Metric listing not deterministic"

    def test_model_registry_determinism(self):
        """Test that model selection is deterministic."""
        # Get passthrough model multiple times
        results = []
        for _ in range(5):
            model_cls = ModelRegistry.get("passthrough")
            results.append(model_cls)

        # All retrievals should return same class
        assert all(r == results[0] for r in results), "Model retrieval not deterministic"
        assert results[0] is not None, "PassThrough model should be registered"
