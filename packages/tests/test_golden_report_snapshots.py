"""Golden report snapshot tests - template regression prevention.

These tests capture key sections of generated reports to catch breaking changes
without requiring 100% coverage of the renderer. Focus on high-risk template areas
that customers see directly.
"""

import pytest


class TestGoldenReportSnapshots:
    """Test report generation critical paths that prevent customer-visible regressions."""

    def test_standard_audit_template_has_required_sections(self):
        """Standard audit template must contain required sections for regulatory compliance."""
        try:
            from importlib.resources import files
        except ImportError:
            pytest.skip("importlib.resources not available")

        template_files = files("glassalpha.report.templates")
        template = template_files.joinpath("standard_audit.html")

        if not template.is_file():
            pytest.fail("standard_audit.html template not found in package")

        content = template.read_text()

        # Critical sections that customers/regulators expect
        required_sections = [
            "<!DOCTYPE html>",  # Valid HTML document
            "<html",  # HTML structure
            "<head>",  # Proper HTML head
            "<title>",  # Document title
            "Model Audit Report",  # Expected title content
            "<body>",  # HTML body
            "Explainability",  # Explainability section
            "Performance",  # Performance metrics section
            "Feature",  # Feature information
        ]

        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)

        assert not missing_sections, f"Template missing required sections: {missing_sections}"

    def test_template_has_valid_html_structure(self):
        """Template must be valid HTML to prevent browser rendering issues."""
        try:
            from importlib.resources import files
        except ImportError:
            pytest.skip("importlib.resources not available")

        template_files = files("glassalpha.report.templates")
        template = template_files.joinpath("standard_audit.html")
        content = template.read_text()

        # Basic HTML validation checks
        assert content.count("<html") == 1, "Must have exactly one <html> tag"
        assert content.count("<head>") == 1, "Must have exactly one <head> tag"
        assert content.count("<body>") == 1, "Must have exactly one <body> tag"
        assert content.count("</html>") == 1, "Must have exactly one </html> closing tag"
        assert content.count("</head>") == 1, "Must have exactly one </head> closing tag"
        assert content.count("</body>") == 1, "Must have exactly one </body> closing tag"

    def test_template_has_jinja2_variables(self):
        """Template must contain Jinja2 variables for dynamic content."""
        try:
            from importlib.resources import files
        except ImportError:
            pytest.skip("importlib.resources not available")

        template_files = files("glassalpha.report.templates")
        template = template_files.joinpath("standard_audit.html")
        content = template.read_text()

        # Key Jinja2 variables that should be present
        expected_variables = [
            "{{ audit_id",  # Audit identification
            "{{ timestamp",  # Report generation time
            "{{ model_info",  # Model information
            "{% for",  # At least one loop construct
            "{% if",  # At least one conditional
        ]

        missing_variables = []
        for var in expected_variables:
            if var not in content:
                missing_variables.append(var)

        assert not missing_variables, f"Template missing Jinja2 variables: {missing_variables}"

    def test_template_css_styling_present(self):
        """Template must include CSS for proper visual formatting."""
        try:
            from importlib.resources import files
        except ImportError:
            pytest.skip("importlib.resources not available")

        template_files = files("glassalpha.report.templates")
        template = template_files.joinpath("standard_audit.html")
        content = template.read_text()

        # Should have some CSS styling
        has_style_tag = "<style>" in content or "<style " in content
        has_css_link = 'rel="stylesheet"' in content
        has_inline_styles = 'style="' in content

        assert has_style_tag or has_css_link or has_inline_styles, (
            "Template should include CSS styling for proper formatting"
        )

    def test_template_size_reasonable(self):
        """Template size should be reasonable - not empty, not massive."""
        try:
            from importlib.resources import files
        except ImportError:
            pytest.skip("importlib.resources not available")

        template_files = files("glassalpha.report.templates")
        template = template_files.joinpath("standard_audit.html")
        content = template.read_text()

        # Size checks
        assert len(content) > 1000, "Template should have substantial content (>1KB)"
        assert len(content) < 1_000_000, "Template should not be massive (<1MB)"

    def test_report_renderer_basic_smoke_test(self):
        """Report renderer must be importable and create basic HTML."""
        try:
            from glassalpha.config.schema import AuditConfig
            from glassalpha.pipeline.audit import AuditResults
            from glassalpha.report.renderer import AuditReportRenderer
        except ImportError as e:
            pytest.skip(f"Report components not available: {e}")

        # Create minimal test data
        config = AuditConfig(
            data_path="dummy.csv",
            model_config={"type": "logistic_regression"},
            audit_profile="tabular_compliance",
            output={"path": "dummy.pdf"},
        )

        # Create minimal audit results
        results = AuditResults(
            audit_id="test-audit-123",
            config=config,
            model_info={},
            explanations={},
            metrics={},
            manifest={},
        )

        # Test renderer can be created
        renderer = AuditReportRenderer()
        assert renderer is not None, "Renderer should be creatable"

        # Test basic rendering doesn't crash
        try:
            html_content = renderer.render_audit_report(results)
            assert isinstance(html_content, str), "Renderer should return string"
            assert len(html_content) > 0, "Rendered content should not be empty"
            assert "html" in html_content.lower(), "Content should contain HTML"
        except Exception as e:
            # If rendering fails due to missing dependencies, that's OK for this smoke test
            # We just want to ensure the basic structure is sound
            if "shap" in str(e).lower() or "matplotlib" in str(e).lower():
                pytest.skip(f"Rendering failed due to missing optional dependency: {e}")
            else:
                raise  # Re-raise unexpected errors

    def test_template_no_security_vulnerabilities(self):
        """Template must not contain potential security vulnerabilities."""
        try:
            from importlib.resources import files
        except ImportError:
            pytest.skip("importlib.resources not available")

        template_files = files("glassalpha.report.templates")
        template = template_files.joinpath("standard_audit.html")
        content = template.read_text()

        # Check for potential security issues
        security_risks = [
            "eval(",  # JavaScript eval
            "document.write",  # Unsafe DOM manipulation
            "innerHTML",  # Potential XSS
            "javascript:",  # JavaScript URLs
            "vbscript:",  # VBScript URLs
            "<script src=",  # External scripts (should be avoided)
        ]

        found_risks = []
        for risk in security_risks:
            if risk in content.lower():
                found_risks.append(risk)

        if found_risks:
            pytest.fail(f"Template contains potential security risks: {found_risks}")

    def test_template_file_size_and_encoding(self):
        """Template file must be properly encoded and reasonable size."""
        try:
            from importlib.resources import files
        except ImportError:
            pytest.skip("importlib.resources not available")

        template_files = files("glassalpha.report.templates")
        template = template_files.joinpath("standard_audit.html")

        # Test UTF-8 encoding works correctly
        content = template.read_text(encoding="utf-8")
        assert isinstance(content, str), "Template should be readable as UTF-8"

        # Test that content is reasonable size
        lines = content.split("\n")
        assert len(lines) > 10, "Template should have multiple lines"
        assert len(lines) < 10000, "Template should not be excessively long"
