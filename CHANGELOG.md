# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **E5.1: Basic Intersectional Fairness** (P1 Feature - Sophistication Signal)

  - **Two-way intersectional analysis**: Detects bias at intersections of protected attributes (e.g., gender×race, age×income)
  - **Cartesian product group creation**: Automatically generates all intersectional groups from attribute combinations
  - **Full fairness metrics for each intersectional group**: TPR, FPR, precision, recall, selection rate computed per intersection
  - **Disparity metrics**: Max-min differences and ratios across intersectional groups
  - **Sample size warnings**: Reuses E10 infrastructure (n<10 ERROR, 10≤n<30 WARNING, n≥30 OK)
  - **Bootstrap confidence intervals**: 95% CIs for all intersectional metrics using E10 bootstrap infrastructure
  - **Statistical power analysis**: Power calculations for detecting disparity in each intersectional group
  - **Deterministic computation**: Fully seeded for byte-identical reproducibility
  - **Automatic integration**: Runs alongside group fairness when `data.intersections` specified in config
  - **JSON export**: All intersectional results serializable with nested structure
  - Configuration: `data.intersections: ["gender*race", "age*income"]` in YAML
  - CLI: Intersectional metrics included when intersections configured
  - API: `compute_intersectional_fairness()`, `create_intersectional_groups()`, `parse_intersection_spec()`
  - Test coverage: 23 tests covering parsing, group creation, metrics, CIs, sample warnings, determinism
  - Module: `glassalpha.metrics.fairness.intersectional`
  - Example: German Credit config updated with age×foreign_worker and gender×age intersections

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

- **Config Schema**: Added `data.intersections` field for E5.1 intersectional fairness analysis
  - Validates intersection format: must be `"attr1*attr2"` for two-way intersections
  - Normalizes to lowercase and strips whitespace
  - Example: `intersections: ["gender*race", "age_years*foreign_worker"]`
- **Fairness Metrics Runner**: Enhanced to compute bootstrap CIs, sample size warnings, power analysis, and intersectional metrics
  - New parameter: `intersections` list for specifying intersectional analysis
  - Automatically computes intersectional fairness when intersections specified
  - Results nested under `intersectional` key in output
- **Audit Pipeline**:
  - Now passes seed to fairness runner for deterministic confidence intervals
  - Now passes intersections config to fairness runner for E5.1 analysis
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
