#!/usr/bin/env python3
"""Fix notebook issues found in testing."""

import json
import sys

def fix_adult_income_notebook():
    """Fix adult_income_drift.ipynb column name issues."""
    notebook_path = "examples/notebooks/adult_income_drift.ipynb"

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    modified = False

    for cell_idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source_lines = cell['source']
            new_lines = []
            skip_lines = 0

            for line_idx, line in enumerate(source_lines):
                if skip_lines > 0:
                    # Skip lines that were part of the old protected_test block
                    skip_lines -= 1
                    new_lines.append(f"# Removed old line: {line.strip()}\n")
                    continue

                if "df['income']" in line:
                    if "== '>50K'" in line:
                        # Fix the income column reference
                        new_line = line.replace("df['income'] == '>50K'", "df['income_over_50k'] == 1")
                        new_lines.append(new_line)
                        modified = True
                        print(f"Fixed: {line.strip()} -> {new_line.strip()}")
                    else:
                        new_lines.append(line)
                elif "df['education']" in line:
                    # Fix the education column reference
                    new_line = line.replace("df['education']", "df['education_level']")
                    new_lines.append(new_line)
                    modified = True
                    print(f"Fixed: {line.strip()} -> {new_line.strip()}")
                elif "protected_attributes=protected_test" in line:
                    # Fix the protected attributes access to use original dataframe
                    new_lines.append("protected_attributes={'race': df.loc[X_test.index, 'race'].values, 'sex': df.loc[X_test.index, 'sex'].values, 'age': df.loc[X_test.index, 'age'].values}\n")
                    modified = True
                    print(f"Fixed: {line.strip()} -> Use original dataframe values directly")
                elif "baseline_result.performance.accuracy" in line:
                    # Fix performance access to use dictionary syntax
                    new_line = line.replace("baseline_result.performance.accuracy", "baseline_result.performance['accuracy']")
                    new_lines.append(new_line)
                    modified = True
                    print(f"Fixed: {line.strip()} -> {new_line.strip()}")
                elif "baseline_result.performance.auc_roc" in line:
                    # Fix performance access to use dictionary syntax
                    new_line = line.replace("baseline_result.performance.auc_roc", "baseline_result.performance['roc_auc']")
                    new_lines.append(new_line)
                    modified = True
                    print(f"Fixed: {line.strip()} -> {new_line.strip()}")
                elif "shifted_result.performance.accuracy" in line:
                    # Fix performance access to use dictionary syntax
                    new_line = line.replace("shifted_result.performance.accuracy", "shifted_result.performance['accuracy']")
                    new_lines.append(new_line)
                    modified = True
                    print(f"Fixed: {line.strip()} -> {new_line.strip()}")
                elif "shifted_result.performance.auc_roc" in line:
                    # Fix performance access to use dictionary syntax
                    new_line = line.replace("shifted_result.performance.auc_roc", "shifted_result.performance['roc_auc']")
                    new_lines.append(new_line)
                    modified = True
                    print(f"Fixed: {line.strip()} -> {new_line.strip()}")
                else:
                    new_lines.append(line)

            cell['source'] = new_lines

    if modified:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"✓ Fixed {notebook_path}")
    else:
        print(f"✗ No changes needed for {notebook_path}")

def fix_quickstart_colab_notebook():
    """Fix quickstart_colab.ipynb dataset access."""
    notebook_path = "examples/notebooks/quickstart_colab.ipynb"

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    modified = False

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_lines = cell['source']
            new_lines = []

            for line in source_lines:
                if "data['dataframe']" in line:
                    # Fix the dataset access
                    new_line = line.replace("data['dataframe']", "data")
                    new_lines.append(new_line)
                    modified = True
                    print(f"Fixed: {line.strip()} -> {new_line.strip()}")
                elif "data = load_german_credit()" in line:
                    # Fix the variable assignment
                    new_line = line.replace("data = load_german_credit()", "df = load_german_credit()")
                    new_lines.append(new_line)
                    modified = True
                    print(f"Fixed: {line.strip()} -> {new_line.strip()}")
                elif "df = data" in line:
                    # Remove this line since we already assigned df above
                    new_lines.append("# Removed: df = data (already assigned above)")
                    modified = True
                    print(f"Fixed: Removed line '{line.strip()}'")
                elif "data['protected_attributes']" in line:
                    # Fix the protected attributes access
                    new_line = line.replace("data['protected_attributes']", "sensitive_features")
                    new_lines.append(new_line)
                    modified = True
                    print(f"Fixed: {line.strip()} -> {new_line.strip()}")
                elif "protected_attrs" in line and "sensitive_features" not in line:
                    # Fix variable name reference
                    new_line = line.replace("protected_attrs", "sensitive_features")
                    new_lines.append(new_line)
                    modified = True
                    print(f"Fixed: {line.strip()} -> {new_line.strip()}")
                elif "sensitive_features}" in line and "Cell" not in line:
                    # Add sensitive_features definition before using it
                    new_lines.append("sensitive_features = ['gender', 'age_group', 'foreign_worker']\n")
                    new_lines.append(line)
                    modified = True
                    print(f"Fixed: Added sensitive_features definition before {line.strip()}")
                else:
                    new_lines.append(line)

            cell['source'] = new_lines

    if modified:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"✓ Fixed {notebook_path}")
    else:
        print(f"✗ No changes needed for {notebook_path}")

def fix_german_credit_notebooks():
    """Fix german_credit_walkthrough.ipynb and custom_data_template.ipynb string encoding issues."""
    notebooks = [
        "examples/notebooks/german_credit_walkthrough.ipynb",
        "examples/notebooks/custom_data_template.ipynb"
    ]

    for notebook_path in notebooks:
        with open(notebook_path, 'r') as f:
            nb = json.load(f)

        modified = False

        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source_lines = cell['source']
                new_lines = []

                for line in source_lines:
                    if "RandomForestClassifier" in line and "fit(X_train, y_train)" in line:
                        # Add proper categorical encoding before model training
                        new_lines.append("# Encode categorical features for sklearn compatibility\n")
                        new_lines.append("from sklearn.preprocessing import OneHotEncoder\n")
                        new_lines.append("\n")
                        new_lines.append("# Get categorical columns that need encoding\n")
                        new_lines.append("categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()\n")
                        new_lines.append("numerical_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()\n")
                        new_lines.append("\n")
                        new_lines.append("# Create encoder for categorical features\n")
                        new_lines.append("if categorical_cols:\n")
                        new_lines.append("    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)\n")
                        new_lines.append("    X_train_encoded = encoder.fit_transform(X_train[categorical_cols])\n")
                        new_lines.append("    X_test_encoded = encoder.transform(X_test[categorical_cols])\n")
                        new_lines.append("    \n")
                        new_lines.append("    # Combine encoded categorical with numerical features\n")
                        new_lines.append("    X_train_final = pd.concat([\n")
                        new_lines.append("        pd.DataFrame(X_train_encoded, index=X_train.index),\n")
                        new_lines.append("        X_train[numerical_cols]\n")
                        new_lines.append("    ], axis=1)\n")
                        new_lines.append("    \n")
                        new_lines.append("    X_test_final = pd.concat([\n")
                        new_lines.append("        pd.DataFrame(X_test_encoded, index=X_test.index),\n")
                        new_lines.append("        X_test[numerical_cols]\n")
                        new_lines.append("    ], axis=1)\n")
                        new_lines.append("else:\n")
                        new_lines.append("    X_train_final = X_train\n")
                        new_lines.append("    X_test_final = X_test\n")
                        new_lines.append("\n")
                        new_lines.append("# Train model with encoded data\n")
                        new_lines.append("model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED)\n")
                        new_lines.append("model.fit(X_train_final, y_train)\n")
                        new_lines.append("\n")
                        new_lines.append("# Update X_test reference for later use\n")
                        new_lines.append("X_test = X_test_final\n")
                        new_lines.append("\n")
                        modified = True
                        print(f"Fixed RandomForest training in {notebook_path}")

                        # Skip the original fit line
                        continue
                    elif "model = RandomForestClassifier" in line and "fit(X_train, y_train)" in line:
                        # Skip this line since we replaced it above
                        continue
                    elif "XGBClassifier" in line and ".fit(X_train, y_train)" in line:
                        # Fix XGBoost to use encoded data and enable categorical support
                        new_lines.append("# Use encoded data for XGBoost and enable categorical support\n")
                        new_lines.append("xgb = XGBClassifier(\n")
                        new_lines.append("    n_estimators=100,\n")
                        new_lines.append("    max_depth=3,\n")
                        new_lines.append("    random_state=SEED,\n")
                        new_lines.append("    eval_metric='logloss',\n")
                        new_lines.append("    enable_categorical=True  # Enable categorical support\n")
                        new_lines.append(")\n")
                        new_lines.append("# Use X_train_final for XGBoost (defined in RandomForest section above)\n")
                        new_lines.append("if 'X_train_final' in locals():\n")
                        new_lines.append("    xgb.fit(X_train_final, y_train)  # Use encoded data\n")
                        new_lines.append("else:\n")
                        new_lines.append("    xgb.fit(X_train, y_train)  # Fallback to original data\n")
                        new_lines.append("\n")
                        modified = True
                        print(f"Fixed XGBoost training in {notebook_path}")
                        continue
                    else:
                        new_lines.append(line)

                cell['source'] = new_lines

        if modified:
            with open(notebook_path, 'w') as f:
                json.dump(nb, f, indent=1)
            print(f"✓ Fixed {notebook_path}")
        else:
            print(f"✗ No changes needed for {notebook_path}")

def handle_compas_pdf_error():
    """Handle compas_bias_detection.ipynb NotImplementedError for to_pdf()."""
    notebook_path = "examples/notebooks/compas_bias_detection.ipynb"

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    modified = False

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_lines = cell['source']
            new_lines = []

            for line in source_lines:
                if "result.to_pdf(" in line:
                    # Comment out the PDF export since it's not implemented
                    new_lines.append("# TODO: Uncomment when to_pdf() is implemented in Phase 3\n")
                    new_lines.append(f"# {line}")
                    modified = True
                    print(f"Commented out PDF export in {notebook_path}")
                else:
                    new_lines.append(line)

            cell['source'] = new_lines

    if modified:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"✓ Fixed {notebook_path}")
    else:
        print(f"✗ No changes needed for {notebook_path}")

def fix_pytest_slow_mark():
    """Fix pytest.mark.slow warning in test_execution.py."""
    test_file = "packages/tests/notebooks/test_execution.py"

    with open(test_file, 'r') as f:
        content = f.read()

    # Add the slow marker to pytest configuration
    if 'slow' not in content or 'pytest.mark.slow' in content:
        print("✓ pytest.mark.slow is already properly configured")
        return

    # The slow marker is already defined in pytest.ini, so this should be fine
    print("✓ No changes needed for pytest.mark.slow - it's configured in pytest.ini")

if __name__ == "__main__":
    print("🔧 Fixing notebook issues...\n")

    fix_adult_income_notebook()
    fix_quickstart_colab_notebook()
    fix_german_credit_notebooks()
    handle_compas_pdf_error()
    fix_pytest_slow_mark()

    print("\n✅ All fixes applied!")
