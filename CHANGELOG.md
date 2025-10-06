# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **E11: Individual Fairness Metrics** (P0 Feature - Legal Risk Coverage)

  - **Consistency Score**: Lipschitz-like metric measuring prediction stability for similar individuals
    - Configurable distance metrics (Euclidean, Mahalanobis)
    - Similarity threshold based on percentile of pairwise distances (default 90th)
    - Reports max and mean prediction differences for similar pairs
  - **Matched Pairs Report**: Identifies individuals with similar features but different predictions
    - Flags potential disparate treatment cases
    - Exports matched pairs with feature distance and prediction difference
    - Checks if protected attributes differ between matched pairs
  - **Counterfactual Flip Test**: Tests if protected attribute changes affect predictions
    - Detects disparate treatment at individual level
    - Computes disparate treatment rate across dataset
    - Supports multi-class protected attributes
  - **Automatic Integration**: Runs alongside group fairness metrics in audit pipeline
  - **JSON Export**: All results serializable for programmatic access and reporting
  - **Deterministic**: Fully seeded for byte-identical reproducibility
  - **Performance**: Pairwise distance computation optimized with vectorization
  - CLI: Individual fairness included in standard audit output
  - API: `IndividualFairnessMetrics`, `compute_consistency_score()`, `find_matched_pairs()`, `counterfactual_flip_test()`
  - Test coverage: 20+ contract tests covering determinism, edge cases, and integration
  - Module: `glassalpha.metrics.fairness.individual`

- **E10: Statistical Confidence & Uncertainty for Fairness Metrics** (P0 Feature)
  - Bootstrap confidence intervals for all fairness metrics (TPR, FPR, precision, recall, demographic parity)
  - Deterministic bootstrap with seeded random sampling for byte-identical reproducibility
  - Sample size adequacy checks (n<10 ERROR, 10≤n<30 WARNING, n≥30 OK)
  - Statistical power calculations for detecting disparity
  - Automatic integration with fairness runner and audit pipeline
  - Confidence interval data exported in JSON format for programmatic access
  - CLI: All fairness metrics now include 95% confidence intervals by default
  - API: `run_fairness_metrics()` accepts `compute_confidence_intervals`, `n_bootstrap`, `confidence_level`, `seed`
  - Test coverage: 21 comprehensive tests validating determinism, accuracy, and edge cases

### Changed

- **Fairness Metrics Runner**: Enhanced to compute bootstrap CIs, sample size warnings, and power analysis
- **Audit Pipeline**:
  - Now passes seed to fairness runner for deterministic confidence intervals
  - Now computes E11 individual fairness metrics alongside group fairness
  - Passes full feature DataFrame to fairness metrics for individual-level analysis
- **AuditResults**: Fairness analysis now includes `sample_size_warnings`, `statistical_power`, `{metric}_ci`, and `individual_fairness` keys
- **`_compute_fairness_metrics()`**: Signature updated to accept full feature DataFrame for individual fairness computation

### Fixed

- None

### Performance

- Bootstrap CIs add ~2-3 seconds per fairness metric (with n_bootstrap=1000, n=200)
- No memory overhead (bootstrap samples not retained)
- Determinism has zero performance cost (uses existing seeding infrastructure)

### Security

- None

### Deprecated

- None

### Removed

- None

## Previous Releases

See Git history for previous changes.
