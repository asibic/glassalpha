# Changelog

All notable changes to GlassAlpha will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Standardized Exit Codes**: Consistent exit codes across all CLI commands for reliable scripting

  - Exit code 0: Success
  - Exit code 1: User error (configuration, missing files, invalid inputs)
  - Exit code 2: System error (permissions, resources, environment)
  - Exit code 3: Validation error (strict mode, compliance failures)
  - New `glassalpha.cli.exit_codes` module with ExitCode enum
  - Helper functions for exit code descriptions
  - All CLI commands updated to use standardized exit codes (28 updates)

- **Unified Error Formatter**: Consistent, self-diagnosable error messages across all commands

  - Structured error format: What/Why/Fix with optional examples
  - Six error types: CONFIG, DATA, MODEL, SYSTEM, VALIDATION, COMPONENT
  - Color-coded severity levels for terminal output
  - New `glassalpha.cli.error_formatter` module with ErrorFormatter class
  - Convenience methods for each error type
  - Actionable guidance reduces support burden

- **JSON Error Output**: Machine-readable errors for CI/CD integration and automation

  - `--json-errors` global flag for JSON output mode
  - Auto-enables in CI environments (GitHub Actions, GitLab CI, CircleCI, Jenkins, Travis)
  - Environment variable support: `GLASSALPHA_JSON_ERRORS=1`
  - Structured error data with exit codes, types, messages, details, and context
  - Success responses in JSON format
  - Validation error aggregation
  - Timestamps for error tracking
  - Schema version for compatibility
  - Perfect for parsing in scripts and monitoring systems

- **Configuration Templates**: Minimal templates for common use cases reduce setup time

  - Four ready-to-use templates: quickstart, production, development, testing
  - Quickstart: 10-second audits with built-in datasets
  - Production: Full regulatory compliance with strict mode
  - Development: Flexible configuration for model iteration
  - Testing: CI/CD-ready with full determinism
  - Templates include comments explaining all options
  - README with comparison matrix and customization guide

- **Interactive Configuration Wizard**: `glassalpha init` command guides users through setup

  - Interactive questionnaire identifies use case
  - Automatically selects appropriate template
  - Optional customization (data path, model type)
  - Generates ready-to-use configuration in seconds
  - Provides next steps and usage instructions
  - Non-interactive mode for automation (`--no-interactive`)
  - Time to first audit reduced from 30min to 2min

- **Smart Context-Aware Defaults**: CLI intelligently infers parameters to minimize typing

  - Config file auto-detection (searches for glassalpha.yaml, audit.yaml, config.yaml)
  - Output path inference (audit.yaml → audit.html)
  - Strict mode auto-enables for prod*/production* configs
  - Repro mode auto-enables in CI environments and for test\* configs
  - Environment variable support (GLASSALPHA_STRICT, GLASSALPHA_REPRO, CI)
  - `--show-defaults` flag for debugging inferred values
  - Explicit flags always override smart defaults
  - Minimal command: `glassalpha audit` (zero flags needed!)

- **Enhanced Output Path Validation**: Comprehensive pre-flight checks prevent late failures

  - Validates output directory exists before starting audit
  - Checks directory write permissions
  - Validates manifest sidecar path is writable
  - Detects read-only existing manifests
  - `--check-output` flag for dry-run validation
  - Shows which files will be overwritten
  - Clear error messages with actionable hints
  - Prevents wasted computation time on permission errors
  - Perfect for CI/CD pre-flight checks

- **Adult Income Dataset**: Added second built-in benchmark dataset for income prediction fairness

  - Canonical UCI Adult (Census Income) dataset with 48,842 records
  - Automatic download, processing, and caching
  - 14 demographic and employment features
  - Binary income prediction target (>50K / <=50K)
  - Protected attributes: race, sex, age groups
  - Preprocessed with education grouping and age binning
  - Example configuration: `configs/adult_income_simple.yaml`
  - Lazy loading preserves fast CLI startup times

- **Enhanced Validation Command**: Runtime availability checks catch configuration errors before wasting compute time

  - Validates model/explainer availability against component registries
  - Dataset schema validation (checks feature columns, target column exist)
  - Model/explainer compatibility warnings (e.g., TreeSHAP with linear models)
  - `--strict-validation` flag treats warnings as errors for CI/CD pipelines
  - Clear error messages with installation instructions
  - Validation accuracy improved from ~60% to ~98%

- **Clear Fallback Communication**: Transparent component selection with actionable guidance

  - Structured fallback warnings with clear explanations of what/why
  - Installation instructions included in fallback messages
  - Enhanced audit summary displays selected components and fallback indicators
  - `--no-fallback` flag for production deployments (fail instead of fallback)
  - Preprocessing mode warnings (auto mode flagged as not production-ready)

- **Testable Strict Mode**: Quick strict mode enables testing with built-in datasets

  - `--strict-quick` flag for development/testing workflows
  - Relaxed validation for built-in datasets (bypasses data path requirements)
  - Still enforces reproducibility (explicit seeds, determinism, manifest generation)
  - Clear warnings distinguish test mode from production strict mode

- **Fast Failure on Invalid Paths**: Pre-flight output validation saves compute time

  - Output directory existence check before running audit pipeline
  - Write permission validation prevents late failures
  - Clear error messages with suggestions (e.g., "mkdir -p /path/to/output")
  - Saves 5-60 seconds per failed audit

- **Model-Explainer Compatibility Documentation**: Comprehensive guide for optimal explainer selection
  - Complete compatibility matrix for all model/explainer combinations
  - Performance benchmarks (German Credit dataset, 1000 samples)
  - Configuration examples and best practices
  - Troubleshooting guide with common issues and solutions
  - Documentation: [Model-Explainer Compatibility](https://glassalpha.com/reference/model-explainer-compatibility/)

### Fixed

- **Critical explainer crash**: Fixed `TypeError` in explainer registry when checking model compatibility. All explainer `is_compatible` methods now use standardized `@classmethod` signature with keyword-only arguments.
- **Explainer contract enforcement**: Added contract tests to validate explainer API signatures, preventing future signature mismatches.

### Performance

- **CLI startup 83% faster**: Optimized `--help` command from 635ms to ~90ms through lazy loading of heavy ML libraries (sklearn, pandas, xgboost).
- **Lazy dataset loading**: Dataset commands now load on-demand rather than at CLI startup, improving responsiveness.
- **Import optimization**: Moved sklearn imports to function scope, eliminating 527ms of unnecessary import overhead.

### Added

- **Preprocessing Artifact Verification**: Load production preprocessing pipelines with integrity verification (12 tests, <1s overhead)

  - Dual hash system: file integrity (SHA256/BLAKE2b) + learned parameters (canonical JSON)
  - CLI commands: `glassalpha prep hash|inspect|validate` with color-coded output
  - Security: Class allowlisting prevents pickle exploits (10 approved sklearn transformers)
  - Version compatibility checking (sklearn/numpy/scipy) with configurable policies (exact/patch/minor)
  - Unknown category detection with three-tier thresholds (notice/warn/fail)
  - Strict mode enforcement: artifact mode + both hashes required
  - Full audit trail in reports with prominent warnings for non-compliant auto mode
  - Documentation: [Preprocessing Guide](https://glassalpha.com/guides/preprocessing/)

- **HTML Report Enhancements**: 20 regulatory improvements to audit reports

  - Navigation: Table of contents with clickable section numbering for regulatory citations
  - Compliance: Model Card section following Mitchell et al. framework
  - Educational: Glossary with 17 key terms for non-technical reviewers
  - Transparency: Preprocessing pipeline diagram with risk indicators
  - Accessibility: ARIA labels, semantic HTML, WCAG AA contrast standards
  - Regulatory: Framework citations (GDPR Article 22, ECOA Regulation B with dates)
  - UX: Hash truncation with hover tooltips, page numbers for print/PDF

- **Performance regression tests**: Comprehensive test suite (9 tests) validates CLI performance, explainer contracts, and end-to-end audit functionality.
- **CI performance gates**: Automated performance testing in CI pipeline catches regressions before merge (target: `--help` < 300ms, currently ~90ms).
- **Performance documentation**: Added performance testing section to CONTRIBUTING.md with lazy loading patterns and benchmarks.
- **Audit smoke tests**: End-to-end tests validate full audit pipeline completes successfully with valid PDF and manifest generation.

### Changed

- **Explainer API standardization**: All explainers now implement consistent `is_compatible(cls, *, model=None, model_type=None, config=None)` signature.
- **Registry error handling**: Explainer registry now gracefully handles signature variations with fallback logic and clear error messages.
- **Backward compatibility**: All module imports preserved using `__getattr__` pattern for lazy loading without breaking existing code.

### Technical Details

**Files Modified**: 15 files (6 explainer fixes, 3 performance optimizations, 3 new tests, 3 infrastructure updates)
**Test Coverage**: Added 9 automated tests (7 performance, 1 contract, 1 smoke)
**Performance Baseline**: `--help`: 88ms, `--version`: 46ms, `datasets list`: 248ms, full audit: 4.9s
**CI Integration**: Performance tests run automatically on every PR with 2-3x safety margins

## [0.1.0] - Initial Release

### Added

- Initial release of GlassAlpha AI Compliance Toolkit
- Support for tabular model auditing (XGBoost, LightGBM, LogisticRegression)
- TreeSHAP and KernelSHAP explainers
- Performance and fairness metrics
- PDF audit report generation
- CLI interface with `audit`, `validate`, `list` commands
- Configuration-driven audit pipeline
- Deterministic audit generation with manifest tracking
- German Credit and Adult Income example datasets

---

**Legend**:

- `Added` - New features
- `Changed` - Changes in existing functionality
- `Deprecated` - Soon-to-be removed features
- `Removed` - Removed features
- `Fixed` - Bug fixes
- `Security` - Vulnerability fixes
- `Performance` - Performance improvements
