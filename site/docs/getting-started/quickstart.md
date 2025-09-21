# Hello Audit Tutorial

Get your first regulator-ready PDF audit report in under 5 minutes.

## Prerequisites
- Python 3.11+
- pip

## Step 1: Install Glass Alpha

```bash
pip install glassalpha
glassalpha --version  # Verify installation
```

## Step 2: Download Sample Data

```bash
# Sample dataset - regulatory compliance benchmark
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
```

## Step 3: Create Your First Audit

```bash
# Generate audit with default config
glassalpha audit --data german.data --target default --out my_first_audit.pdf

# ✅ Done! Open my_first_audit.pdf to see your audit report
```

## Step 4: Verify Determinism

```bash
# Run the same audit again
glassalpha audit --data german.data --target default --out my_second_audit.pdf

# Compare - should be byte-identical
diff my_first_audit.pdf my_second_audit.pdf
# (no output = identical files)
```

## What You Just Created

Your PDF audit report includes:

1. **Executive Summary** - Key metrics and findings
2. **Model Performance** - Accuracy, confusion matrix, ROC curves  
3. **Feature Importance** - TreeSHAP explanations and rankings
4. **Fairness Analysis** - Basic group parity metrics
5. **Reproducibility Manifest** - Seeds, hashes, git commit for auditability

## Advanced Configuration

For production use, create a YAML config file:

```yaml
# audit_config.yaml
model:
  type: xgboost
  params:
    max_depth: 5
    learning_rate: 0.1

data:
  train_path: german_train.csv  
  test_path: german_test.csv
  target_column: default

audit:
  protected_attributes:
    - gender
    - age_group
  confidence_level: 0.95
  
reproducibility:
  random_seed: 42
  track_git: true
```

Then run:
```bash
glassalpha audit --config audit_config.yaml --out german_audit_2024.pdf
```



## Next Steps

- 📊 [Financial Credit Deep Dive](../examples/german-credit.md)
- ⚙️ [Configuration Reference](configuration.md)  
- 🏛️ [Regulatory Compliance](../compliance/overview.md)

Remember: Glass Alpha Phase 1 is **audit-first**. One command, regulator-ready PDF.
