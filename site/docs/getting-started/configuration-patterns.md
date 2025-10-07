# Configuration Patterns

Copy-paste configuration recipes for common use cases. Each pattern is production-ready and can be customized for your needs.

## Quick Reference

| Pattern                                                | Use Case          | Strictness | Best For              |
| ------------------------------------------------------ | ----------------- | ---------- | --------------------- |
| [Credit Scoring](#credit-scoring-standard)             | Banking/lending   | Standard   | Development, testing  |
| [Credit Scoring (Strict)](#credit-scoring-strict-mode) | Banking/lending   | Strict     | Regulatory submission |
| [Healthcare Risk](#healthcare-risk-prediction)         | Medical AI        | Standard   | Hospital ML teams     |
| [Fraud Detection](#fraud-detection)                    | Security/payments | Standard   | Anti-fraud teams      |
| [Insurance Underwriting](#insurance-underwriting)      | Insurance         | Standard   | Actuarial teams       |
| [Minimal (Dev)](#minimal-development-config)           | Any               | Minimal    | Quick testing         |
| [Enterprise (CI/CD)](#enterprise-cicd-config)          | Any               | Strict     | Production pipelines  |

---

## Credit Scoring (Standard)

**Use case**: Loan approval, credit line increase decisions

**Regulatory context**: SR 11-7, ECOA/Reg B compliance

```yaml
# credit_scoring_audit.yaml
model:
  path: models/credit_model.joblib
  type: xgboost # or random_forest, lightgbm
  version: "2.1.0"

data:
  test_data: data/credit_test.csv
  target_column: approved
  protected_attributes:
    - gender
    - race
    - age_group
  schema: schemas/credit_schema.yaml # Optional but recommended

random_seed: 42

explainer:
  strategy: first_compatible
  priority:
    - treeshap # Fast for tree models
    - kernelshap # Fallback
  max_samples: 1000

fairness:
  threshold: 0.5
  tolerance: 0.05 # 5% max difference across groups
  min_group_size: 30
  metrics:
    - demographic_parity
    - equal_opportunity
  confidence_level: 0.95

calibration:
  n_bins: 10
  strategy: uniform
  compute_confidence: true

policy:
  gates:
    min_accuracy: 0.70 # SR 11-7 guidance
    max_bias: 0.10 # ECOA threshold
    max_ece: 0.05 # Calibration requirement
  citations:
    min_accuracy: "SR 11-7 Section 4.2"
    max_bias: "ECOA Regulation B § 1002.2"
  fail_on_violation: false # Warning only in dev

output:
  pdf_path: reports/credit_audit_{date}.pdf
  json_path: reports/metrics_{date}.json
  manifest_path: reports/manifest_{date}.json
  template: financial_services

metadata:
  project: "Credit Model Q1 2025"
  auditor: "Risk Team"
  regulatory_submission: false
```

**To run:**

```bash
glassalpha audit --config credit_scoring_audit.yaml --output credit_report.pdf
```

---

## Credit Scoring (Strict Mode)

**Use case**: Regulatory submission, examiner review

**Key differences**: Enforces all reproducibility requirements

```yaml
# credit_scoring_strict.yaml
model:
  path: models/credit_model_v2.1.0.joblib
  type: xgboost
  version: "2.1.0" # REQUIRED in strict mode
  preprocessing:
    path: preprocessing/pipeline_v2.1.0.joblib

data:
  test_data: data/credit_test_2025_q1.csv
  target_column: approved
  protected_attributes:
    - gender
    - race
    - age_group
  schema: schemas/credit_schema_v1.yaml # REQUIRED in strict mode
  index_column: application_id

random_seed: 42 # REQUIRED in strict mode

explainer:
  strategy: first_compatible
  priority:
    - treeshap
  max_samples: 1000 # Fixed for reproducibility

fairness:
  threshold: 0.5
  tolerance: 0.05
  min_group_size: 50 # Higher threshold for stability
  metrics:
    - demographic_parity
    - equal_opportunity
    - equalized_odds # All three for completeness
  confidence_level: 0.95

calibration:
  n_bins: 10
  strategy: uniform
  compute_confidence: true
  confidence_level: 0.95

policy:
  gates:
    min_accuracy: 0.70
    max_bias: 0.10
    max_ece: 0.05
    min_auc: 0.75
  citations:
    min_accuracy: "SR 11-7 Section 4.2"
    max_bias: "ECOA Regulation B § 1002.2"
    max_ece: "Internal Policy v2.1"
    min_auc: "SR 11-7 Section 4.1"
  fail_on_violation: true # Fail CI/CD if gates violated

output:
  pdf_path: regulatory_submissions/credit_audit_2025_q1.pdf
  json_path: regulatory_submissions/metrics_2025_q1.json
  manifest_path: regulatory_submissions/manifest_2025_q1.json
  template: financial_services
  include_plots: true

metadata:
  project: "Credit Model Q1 2025 Regulatory Audit"
  auditor: "Jane Smith, Model Risk Manager"
  department: "Model Risk Management"
  regulatory_submission: true
  submission_date: "2025-03-15"
  examiner: "Federal Reserve Bank"
```

**To run with strict validation:**

```bash
glassalpha audit --config credit_scoring_strict.yaml --output report.pdf --strict
```

---

## Healthcare Risk Prediction

**Use case**: Patient readmission, treatment recommendations

**Regulatory context**: HIPAA compliance, clinical validation

```yaml
# healthcare_audit.yaml
model:
  path: models/readmission_model.joblib
  type: random_forest
  version: "1.3.0"

data:
  test_data: data/patient_test_deidentified.csv # Must be de-identified
  target_column: readmitted_30d
  protected_attributes:
    - race
    - ethnicity
    - age_group
    - sex
  schema: schemas/patient_schema.yaml

random_seed: 42

explainer:
  strategy: first_compatible
  priority:
    - treeshap
    - kernelshap
  max_samples: 500 # Reduce if dataset is large

fairness:
  threshold: 0.5
  tolerance: 0.05 # Strict for healthcare equity
  min_group_size: 30
  metrics:
    - demographic_parity
    - equal_opportunity # Critical for healthcare access
  confidence_level: 0.95

calibration:
  n_bins: 10
  strategy: quantile # Better for imbalanced healthcare data
  compute_confidence: true

policy:
  gates:
    min_accuracy: 0.75 # Higher bar for clinical use
    max_bias: 0.05 # Strict healthcare equity requirement
    max_ece: 0.05
    min_recall: 0.80 # Don't miss high-risk patients
  citations:
    min_accuracy: "Clinical Validation Protocol v1.2"
    max_bias: "Hospital Equity Policy"
    min_recall: "Patient Safety Guidelines"
  fail_on_violation: true

output:
  pdf_path: clinical_audits/readmission_audit_{date}.pdf
  json_path: clinical_audits/metrics_{date}.json
  template: healthcare

metadata:
  project: "30-Day Readmission Model"
  clinical_lead: "Dr. Sarah Johnson"
  department: "Clinical Informatics"
  irb_approval: "IRB-2024-001"
```

---

## Fraud Detection

**Use case**: Transaction fraud, account takeover detection

**Key characteristics**: High precision, low latency, real-time monitoring

```yaml
# fraud_detection_audit.yaml
model:
  path: models/fraud_detector.joblib
  type: xgboost
  version: "3.2.0"

data:
  test_data: data/transactions_test.csv
  target_column: is_fraud
  protected_attributes:
    - customer_segment
    - geography_region
  schema: schemas/transaction_schema.yaml

random_seed: 42

explainer:
  strategy: fastest # Prioritize speed for real-time use
  priority:
    - treeshap
  max_samples: 500 # Reduce for faster computation
  features_to_explain: 10 # Limit for performance

fairness:
  threshold: 0.7 # High threshold for fraud (minimize false positives)
  tolerance: 0.10 # More permissive for fraud use case
  min_group_size: 50
  metrics:
    - demographic_parity # Ensure no systematic bias
  confidence_level: 0.90

calibration:
  n_bins: 10
  strategy: uniform
  compute_confidence: true

policy:
  gates:
    min_precision: 0.90 # Minimize false positives
    min_recall: 0.70 # Catch most fraud
    max_bias: 0.15 # Relaxed for fraud detection
    max_ece: 0.10
  citations:
    min_precision: "Fraud Policy v2.0"
    min_recall: "Risk Appetite Statement"
  fail_on_violation: false # Warning only

output:
  pdf_path: fraud_audits/fraud_model_audit.pdf
  json_path: fraud_audits/metrics.json
  template: fraud_detection

metadata:
  project: "Transaction Fraud Detection v3"
  team: "Fraud Prevention"
  deployment_env: "production"
```

---

## Insurance Underwriting

**Use case**: Policy pricing, risk assessment

**Regulatory context**: NAIC guidelines, state insurance regulations

```yaml
# insurance_audit.yaml
model:
  path: models/underwriting_model.joblib
  type: lightgbm
  version: "1.5.0"

data:
  test_data: data/applications_test.csv
  target_column: risk_score
  protected_attributes:
    - gender
    - age_group
    - marital_status
    - zip_code # Proxy for race/location
  schema: schemas/insurance_schema.yaml

random_seed: 42

explainer:
  strategy: first_compatible
  priority:
    - treeshap
    - kernelshap
  max_samples: 1000

fairness:
  threshold: 0.5
  tolerance: 0.08 # NAIC guidance
  min_group_size: 40
  metrics:
    - demographic_parity
    - equal_opportunity
  confidence_level: 0.95

calibration:
  n_bins: 10
  strategy: uniform
  compute_confidence: true

policy:
  gates:
    min_accuracy: 0.70
    max_bias: 0.10
    max_ece: 0.05
  citations:
    max_bias: "NAIC Model Act Section 5"
  fail_on_violation: true

output:
  pdf_path: regulatory/insurance_audit_{state}_{date}.pdf
  json_path: regulatory/metrics_{state}_{date}.json
  template: insurance

metadata:
  project: "Auto Insurance Underwriting"
  state: "California"
  filing_id: "CA-2025-001"
  actuary: "John Doe, FSA"
```

---

## Minimal (Development Config)

**Use case**: Quick testing, iterative development

**Key characteristics**: Minimal settings, fast execution

```yaml
# minimal_dev.yaml
model:
  path: models/dev_model.joblib
  type: random_forest

data:
  test_data: data/test_sample.csv
  target_column: target

random_seed: 42
# All other settings use defaults
# Fast iteration, no PDF generation needed
```

**To run:**

```bash
glassalpha audit --config minimal_dev.yaml --output metrics.json
```

---

## Enterprise CI/CD Config

**Use case**: Automated testing in production pipelines

**Key characteristics**: Strict validation, policy gates, CI/CD integration

```yaml
# enterprise_ci.yaml
model:
  path: ${MODEL_PATH} # Environment variable
  type: ${MODEL_TYPE}
  version: ${MODEL_VERSION}
  preprocessing:
    path: ${PREPROCESSING_PATH}

data:
  test_data: ${TEST_DATA_PATH}
  target_column: ${TARGET_COLUMN}
  protected_attributes: ${PROTECTED_ATTRS} # JSON array
  schema: ${SCHEMA_PATH}

random_seed: 42

explainer:
  strategy: first_compatible
  priority:
    - treeshap
  max_samples: 1000

fairness:
  threshold: 0.5
  tolerance: 0.05
  min_group_size: 30
  metrics:
    - demographic_parity
    - equal_opportunity
  confidence_level: 0.95

calibration:
  n_bins: 10
  compute_confidence: true

policy:
  gates:
    min_accuracy: ${MIN_ACCURACY}
    max_bias: ${MAX_BIAS}
    max_ece: ${MAX_ECE}
  fail_on_violation: true # Fail CI/CD if violated

output:
  pdf_path: ${CI_ARTIFACTS_DIR}/audit_${CI_COMMIT_SHA}.pdf
  json_path: ${CI_ARTIFACTS_DIR}/metrics_${CI_COMMIT_SHA}.json
  manifest_path: ${CI_ARTIFACTS_DIR}/manifest_${CI_COMMIT_SHA}.json

metadata:
  ci_job: ${CI_JOB_ID}
  git_commit: ${CI_COMMIT_SHA}
  git_branch: ${CI_BRANCH}
  build_timestamp: ${CI_TIMESTAMP}
```

**GitHub Actions example:**

```yaml
name: ML Audit

on: [pull_request]

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run Audit
        env:
          MODEL_PATH: "models/production_model.joblib"
          MODEL_TYPE: "xgboost"
          TEST_DATA_PATH: "data/test.csv"
          TARGET_COLUMN: "approved"
          MIN_ACCURACY: "0.70"
          MAX_BIAS: "0.10"
        run: |
          pip install glassalpha[explain]
          glassalpha audit --config enterprise_ci.yaml --strict

      - name: Upload Audit Report
        uses: actions/upload-artifact@v3
        with:
          name: audit-report
          path: artifacts/audit_*.pdf
```

---

## Customization Guide

### Common Adjustments

**Fairness tolerance:**

- Banking: `0.05` (5%) - strict
- Healthcare: `0.05` (5%) - strict
- Insurance: `0.08` (8%) - moderate
- Fraud: `0.10-0.15` (10-15%) - relaxed

**Decision threshold:**

- Credit approval: `0.5` (balanced)
- Fraud detection: `0.7-0.8` (high precision)
- Healthcare risk: `0.3-0.4` (high recall, don't miss cases)

**Explainer samples:**

- Development: `100-500` (fast iteration)
- Production: `1000` (balanced)
- Regulatory: `1000+` (maximum accuracy)

### Environment Variables

For CI/CD, use environment variables:

```bash
export MODEL_PATH="models/my_model.joblib"
export TARGET_COLUMN="approved"
export MIN_ACCURACY="0.70"
```

Then reference in config: `${MODEL_PATH}`

### Multiple Configs

Organize configs by environment:

```
configs/
  dev/
    quick_test.yaml
  staging/
    credit_audit_staging.yaml
  production/
    credit_audit_prod.yaml
    credit_audit_prod_strict.yaml
```

---

## Related Documentation

- **[Configuration Guide](configuration.md)** - Full config reference
- **[Policy Gates](../guides/policy-gates.md)** - Compliance thresholds
- **[Strict Mode](../reference/strict-mode.md)** - Regulatory requirements
- **[ML Engineer Workflow](../guides/ml-engineer-workflow.md)** - CI/CD integration
- **[Banking Compliance](../compliance/banking-guide.md)** - SR 11-7 requirements

## Support

- **GitHub Issues**: [Report problems](https://github.com/GlassAlpha/glassalpha/issues)
- **Discussions**: [Ask questions](https://github.com/GlassAlpha/glassalpha/discussions)
- **Email**: [contact@glassalpha.com](mailto:contact@glassalpha.com)
