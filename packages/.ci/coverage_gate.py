#!/usr/bin/env python3
"""Coverage Gate 2: Trend-based coverage checking for full repository."""

import json
import os
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


def get_current_coverage():
    """Parse coverage.xml to get the total coverage percentage."""
    try:
        # Parse coverage.xml
        if os.path.exists("coverage.xml"):
            tree = ET.parse("coverage.xml")
            root = tree.getroot()

            # Find the total coverage line
            for line in root.iter("coverage"):
                if "line-rate" in line.attrib:
                    line_rate = float(line.attrib["line-rate"])
                    return round(line_rate * 100, 2)

        # Fallback: try to parse from text output if available
        if os.path.exists(".coverage"):
            import coverage

            cov = coverage.Coverage()
            cov.load()
            # Get the total coverage percentage
            total_stats = cov.stats
            if hasattr(total_stats, "total"):
                total = total_stats.total
                if total.numbers:
                    covered, total_lines = total.numbers[0]
                    return round((covered / total_lines) * 100, 2)

    except Exception as e:
        print(f"❌ Could not parse coverage data: {e}")
        return None

    print("❌ Could not find coverage data")
    return None


def load_baseline():
    """Load the baseline coverage from .ci/coverage-baseline.json."""
    baseline_file = Path(".ci/coverage-baseline.json")

    if baseline_file.exists():
        try:
            with open(baseline_file) as f:
                data = json.load(f)
                return data.get("coverage", 0.0)
        except Exception as e:
            print(f"⚠️  Could not load baseline: {e}")

    return None


def save_baseline(coverage):
    """Save the current coverage as the new baseline."""
    baseline_file = Path(".ci/coverage-baseline.json")
    baseline_file.parent.mkdir(exist_ok=True)

    try:
        with open(baseline_file, "w") as f:
            json.dump({"coverage": coverage}, f, indent=2)
        print(f"✅ Updated baseline to {coverage}%")
    except Exception as e:
        print(f"⚠️  Could not save baseline: {e}")


def main():
    """Run Gate 2 coverage checking with trend analysis."""
    current = get_current_coverage()

    if current is None:
        print("❌ Could not determine current coverage")
        return 1

    baseline = load_baseline()

    print(f"Repo coverage: {current}% (baseline: {baseline}%, threshold: 70%)")

    # Check if below 70% threshold
    if current < 70:
        print("⚠️  WARNING: Coverage is below 70% target")
        print("This is acceptable for now, but aim to improve coverage over time.")

    # Check for regression (only if we have a baseline)
    if baseline is not None:
        regression_threshold = 0.1  # 0.1 percentage point
        if current < (baseline - regression_threshold):
            print(f"❌ REGRESSION DETECTED: Coverage dropped from {baseline}% to {current}%")
            print(f"This is a {baseline - current:.2f} percentage point decrease")
            return 1

    # Update baseline on success
    save_baseline(current)
    return 0


if __name__ == "__main__":
    sys.exit(main())
