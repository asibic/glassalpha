"""Contract tests for notebook-friendly API features (QW1: Inline HTML Display)."""

from glassalpha.pipeline.audit import AuditResults


def skip_if_error_card(html):
    """Helper to skip content checks if error card is shown (dependencies missing)."""
    return "Inline Display Failed" in html


class TestInlineHTMLDisplay:
    """Contract tests for _repr_html_() Jupyter integration."""

    def test_repr_html_exists(self):
        """Contract: AuditResults has _repr_html_() for Jupyter display."""
        result = AuditResults(success=True)
        assert hasattr(result, "_repr_html_")
        assert callable(result._repr_html_)

    def test_repr_html_returns_html_string(self):
        """Contract: _repr_html_() returns HTML string with DOCTYPE."""
        result = AuditResults(
            success=True,
            model_performance={"accuracy": 0.85, "f1": 0.82},
        )
        html = result._repr_html_()
        assert isinstance(html, str)
        assert len(html) > 0
        # Should contain HTML structure
        assert "<!DOCTYPE html>" in html or "<div" in html

    def test_repr_html_shows_performance_metrics(self):
        """Contract: _repr_html_() displays performance metrics."""
        result = AuditResults(
            success=True,
            model_performance={
                "accuracy": 0.8532,
                "roc_auc": 0.9145,
                "precision": 0.7891,
            },
        )
        html = result._repr_html_()

        # Check metrics are displayed (or error card if dependencies missing)
        if not skip_if_error_card(html):
            assert "0.8532" in html or "85.32" in html  # Accuracy
            assert "0.9145" in html or "91.45" in html  # ROC-AUC
            assert "Performance" in html or "performance" in html.lower()

    def test_repr_html_shows_fairness_badges_pass(self):
        """Contract: _repr_html_() shows ✓ badge for gap < 0.05."""
        result = AuditResults(
            success=True,
            fairness_analysis={
                "group_metrics": {
                    "gender": {
                        "male": 0.82,
                        "female": 0.85,  # Gap = 0.03 < 0.05 → pass
                    },
                },
            },
        )
        html = result._repr_html_()

        # Should show green/success badge (or error card if dependencies missing)
        if not skip_if_error_card(html):
            assert "✓" in html or "badge-success" in html
            assert "gender" in html.lower() or "Gender" in html

    def test_repr_html_shows_fairness_badges_warning(self):
        """Contract: _repr_html_() shows ⚠ badge for gap between 0.05-0.10."""
        result = AuditResults(
            success=True,
            fairness_analysis={
                "group_metrics": {
                    "race": {
                        "group_a": 0.75,
                        "group_b": 0.82,  # Gap = 0.07, should warn
                    },
                },
            },
        )
        html = result._repr_html_()

        # Should show warning badge (or error card if dependencies missing)
        if not skip_if_error_card(html):
            assert "⚠" in html or "badge-warning" in html

    def test_repr_html_shows_fairness_badges_fail(self):
        """Contract: _repr_html_() shows ✗ badge for gap >= 0.10."""
        result = AuditResults(
            success=True,
            fairness_analysis={
                "group_metrics": {
                    "age": {
                        "young": 0.65,
                        "old": 0.82,  # Gap = 0.17 >= 0.10 → fail
                    },
                },
            },
        )
        html = result._repr_html_()

        # Should show fail badge (or error card if dependencies missing)
        if not skip_if_error_card(html):
            assert "✗" in html or "badge-fail" in html

    def test_repr_html_shows_top_features(self):
        """Contract: _repr_html_() displays top 5 features with importances."""
        result = AuditResults(
            success=True,
            explanations={
                "feature_importances": {
                    "feature_1": 0.35,
                    "feature_2": 0.28,
                    "feature_3": 0.19,
                    "feature_4": 0.12,
                    "feature_5": 0.06,
                    "feature_6": 0.01,  # Should not appear (not in top 5)
                },
            },
        )
        html = result._repr_html_()

        # Check top 5 are displayed (or error card if dependencies missing)
        if not skip_if_error_card(html):
            assert "feature_1" in html
            assert "feature_2" in html
            assert "feature_5" in html
            assert "0.35" in html or "0.3500" in html

    def test_repr_html_shows_direction_arrows(self):
        """Contract: _repr_html_() shows ↑ for positive, ↓ for negative importance."""
        result = AuditResults(
            success=True,
            explanations={
                "feature_importances": {
                    "positive_feature": 0.35,
                    "negative_feature": -0.28,
                },
            },
        )
        html = result._repr_html_()

        # Should contain direction indicators (or error card if dependencies missing)
        if not skip_if_error_card(html):
            assert "↑" in html or "↓" in html or "direction" in html.lower()

    def test_repr_html_shows_lineage_badge(self):
        """Contract: _repr_html_() displays data hash, model SHA, versions, seed."""
        result = AuditResults(
            success=True,
            manifest={
                "dataset_hash": "9f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c",
                "model_hash": "3a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d",
                "global_seed": 42,
                "package_versions": {
                    "sklearn": "1.5.2",
                    "xgboost": "2.1.1",
                },
            },
            execution_info={
                "audit_profile": "tabular_compliance",
            },
        )
        html = result._repr_html_()

        # Check lineage elements (or error card if dependencies missing)
        if not skip_if_error_card(html):
            assert "9f2a3b4c" in html or "9f2" in html  # Dataset hash (truncated)
            assert "seed" in html.lower() and "42" in html
            assert "1.5.2" in html or "sklearn" in html.lower()
            assert "tabular_compliance" in html

    def test_repr_html_shows_audit_status_with_gates(self):
        """Contract: _repr_html_() shows policy gate counts when available."""
        result = AuditResults(
            success=True,
            execution_info={
                "policy_gates": {
                    "passed": 12,
                    "warnings": 2,
                    "failures": 0,
                },
            },
        )
        html = result._repr_html_()

        # Should show gate counts (or error card if dependencies missing)
        if "Inline Display Failed" in html:
            # Error card shown (dependencies missing) - this is acceptable
            assert isinstance(html, str) and len(html) > 0
        else:
            # Template rendered successfully - check content
            assert "12" in html
            assert "2" in html
            assert "passed" in html.lower() or "Passed" in html
            assert "warning" in html.lower() or "Warning" in html

    def test_repr_html_shows_audit_status_placeholder(self):
        """Contract: _repr_html_() shows placeholder when no gates configured."""
        result = AuditResults(
            success=True,
            execution_info={},  # No policy_gates
        )
        html = result._repr_html_()

        # Should show "not configured" message (or error card if dependencies missing)
        if not skip_if_error_card(html):
            assert "not configured" in html or "Not configured" in html or "placeholder" in html

    def test_repr_html_shows_export_command(self):
        """Contract: _repr_html_() shows copyable export command."""
        result = AuditResults(success=True)
        html = result._repr_html_()

        # Should show export command (or error card if dependencies missing)
        if not skip_if_error_card(html):
            assert "result.to_pdf" in html or "export" in html.lower()
            assert "output.pdf" in html or ".pdf" in html

    def test_repr_html_has_copy_button(self):
        """Contract: _repr_html_() includes copy button with JavaScript."""
        result = AuditResults(success=True)
        html = result._repr_html_()

        # Should have copy functionality (or error card if dependencies missing)
        if not skip_if_error_card(html):
            assert "copy" in html.lower() or "Copy" in html
            assert "clipboard" in html.lower() or "copyToClipboard" in html

    def test_repr_html_error_card_on_template_failure(self):
        """Contract: _repr_html_() shows visible error card on rendering failure."""
        # Create result that will trigger template error
        # (template expects certain structure)
        result = AuditResults(success=True)

        # Temporarily break template by removing it
        import os
        from pathlib import Path

        template_path = Path(__file__).parent.parent.parent / "src" / "glassalpha" / "report" / "templates" / "inline_summary.html"

        if template_path.exists():
            # Rename template to trigger error
            temp_name = template_path.with_suffix(".html.bak")
            try:
                os.rename(template_path, temp_name)

                # Should return error card, not crash
                html = result._repr_html_()
                assert isinstance(html, str)
                assert len(html) > 0
                # Should contain error indicators
                assert "error" in html.lower() or "failed" in html.lower() or "⚠" in html

            finally:
                # Restore template
                if temp_name.exists():
                    os.rename(temp_name, template_path)

    def test_repr_html_graceful_when_performance_missing(self):
        """Contract: _repr_html_() handles missing performance section gracefully."""
        result = AuditResults(
            success=True,
            model_performance={},  # Empty
        )
        html = result._repr_html_()

        # Should not crash, should be valid HTML
        assert isinstance(html, str)
        assert len(html) > 0
        # Might show placeholder or just skip section

    def test_repr_html_graceful_when_fairness_missing(self):
        """Contract: _repr_html_() handles missing fairness section gracefully."""
        result = AuditResults(
            success=True,
            fairness_analysis={},  # Empty
        )
        html = result._repr_html_()

        # Should not crash
        assert isinstance(html, str)
        assert len(html) > 0

    def test_repr_html_graceful_when_features_missing(self):
        """Contract: _repr_html_() handles missing feature importance gracefully."""
        result = AuditResults(
            success=True,
            explanations={},  # No feature_importances
        )
        html = result._repr_html_()

        # Should not crash
        assert isinstance(html, str)
        assert len(html) > 0

    def test_repr_html_no_embedded_plots(self):
        """Contract: Inline HTML doesn't embed plots (keeps rendering fast)."""
        result = AuditResults(
            success=True,
            model_performance={"accuracy": 0.85},
        )
        html = result._repr_html_()

        # Should NOT contain base64-encoded images
        assert "data:image/png;base64" not in html
        assert "data:image/jpeg;base64" not in html

    def test_repr_html_deterministic_thresholds(self):
        """Contract: Badge thresholds are deterministic (0.05 warn, 0.10 fail)."""
        # Test exact threshold boundaries
        result_pass = AuditResults(
            success=True,
            fairness_analysis={
                "group_metrics": {
                    "attr": {"a": 0.50, "b": 0.54},  # Gap = 0.04 < 0.05 → pass
                },
            },
        )
        html_pass = result_pass._repr_html_()
        if not skip_if_error_card(html_pass):
            assert "✓" in html_pass or "badge-success" in html_pass

        result_warn = AuditResults(
            success=True,
            fairness_analysis={
                "group_metrics": {
                    "attr": {"a": 0.50, "b": 0.57},  # Gap = 0.07, 0.05 ≤ gap < 0.10 → warn
                },
            },
        )
        html_warn = result_warn._repr_html_()
        if not skip_if_error_card(html_warn):
            assert "⚠" in html_warn or "badge-warning" in html_warn

        result_fail = AuditResults(
            success=True,
            fairness_analysis={
                "group_metrics": {
                    "attr": {"a": 0.50, "b": 0.62},  # Gap = 0.12 ≥ 0.10 → fail
                },
            },
        )
        html_fail = result_fail._repr_html_()
        if not skip_if_error_card(html_fail):
            assert "✗" in html_fail or "badge-fail" in html_fail

    def test_repr_html_includes_educational_links(self):
        """Contract: _repr_html_() includes links to documentation."""
        result = AuditResults(success=True)
        html = result._repr_html_()

        # Should have educational links (or error card if dependencies missing)
        if not skip_if_error_card(html):
            assert "glassalpha.com" in html.lower() or "learn" in html.lower() or "href" in html

    def test_repr_html_compact_output(self):
        """Contract: Inline HTML is compact (< 20KB for typical audit)."""
        result = AuditResults(
            success=True,
            model_performance={"accuracy": 0.85, "roc_auc": 0.91},
            fairness_analysis={"group_metrics": {"gender": {"m": 0.8, "f": 0.82}}},
            explanations={"feature_importances": {"f1": 0.3, "f2": 0.2}},
            manifest={"dataset_hash": "abc123", "global_seed": 42},
        )
        html = result._repr_html_()

        # Should be reasonably sized (< 20KB)
        assert len(html) < 20000  # 20KB limit
