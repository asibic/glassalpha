# Changelog

All notable changes to GlassAlpha will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Critical explainer crash**: Fixed `TypeError` in explainer registry when checking model compatibility. All explainer `is_compatible` methods now use standardized `@classmethod` signature with keyword-only arguments.
- **Explainer contract enforcement**: Added contract tests to validate explainer API signatures, preventing future signature mismatches.

### Performance

- **CLI startup 83% faster**: Optimized `--help` command from 635ms to ~90ms through lazy loading of heavy ML libraries (sklearn, pandas, xgboost).
- **Lazy dataset loading**: Dataset commands now load on-demand rather than at CLI startup, improving responsiveness.
- **Import optimization**: Moved sklearn imports to function scope, eliminating 527ms of unnecessary import overhead.

### Added

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
