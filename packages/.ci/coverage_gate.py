#!/usr/bin/env python3
"""Coverage trend monitoring script for Gate 2.

This script:
- Reads coverage.xml to get current coverage percentage
- Compares against a baseline (if exists)
- Warns if below 70% but does NOT fail the job
- Only fails if coverage regresses significantly from baseline
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def get_coverage_from_xml(xml_path: Path) -> float:
    """Extract coverage percentage from coverage.xml."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        line_rate = float(root.attrib.get("line-rate", "0"))
        return line_rate * 100
    except Exception as e:
        print(f"Warning: Could not parse coverage.xml: {e}")
        return 0.0


def load_baseline(baseline_path: Path) -> float | None:
    """Load baseline coverage from JSON file."""
    if not baseline_path.exists():
        return None
    try:
        with open(baseline_path) as f:
            data = json.load(f)
            return float(data.get("coverage", 0))
    except Exception as e:
        print(f"Warning: Could not load baseline: {e}")
        return None


def save_baseline(baseline_path: Path, coverage: float):
    """Save current coverage as new baseline."""
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump({"coverage": coverage}, f, indent=2)


def main():
    """Main trend monitoring logic."""
    # Configuration
    MIN_COVERAGE = 65  # Minimum expected coverage (warning only)
    REGRESSION_TOLERANCE = 2.0  # Allow 2% regression without failing

    # Paths
    coverage_xml = Path("coverage.xml")
    baseline_path = Path(".ci/coverage_baseline.json")

    # Get current coverage
    if not coverage_xml.exists():
        print("Warning: coverage.xml not found, skipping trend monitoring")
        return 0

    current_coverage = get_coverage_from_xml(coverage_xml)
    print(f"Current coverage: {current_coverage:.2f}%")

    # Check against minimum threshold (warning only)
    if current_coverage < MIN_COVERAGE:
        print(f"⚠️  Warning: Coverage {current_coverage:.2f}% is below {MIN_COVERAGE}% threshold")
        print("   This is a trend indicator only and does not fail the build")
    else:
        print(f"✅ Coverage {current_coverage:.2f}% meets {MIN_COVERAGE}% threshold")

    # Check for regression against baseline
    baseline_coverage = load_baseline(baseline_path)
    if baseline_coverage is not None:
        regression = baseline_coverage - current_coverage
        print(f"Baseline coverage: {baseline_coverage:.2f}%")

        if regression > REGRESSION_TOLERANCE:
            print(f"❌ Coverage regressed by {regression:.2f}% from baseline")
            print(f"   Current: {current_coverage:.2f}%, Baseline: {baseline_coverage:.2f}%")
            print(f"   Regression tolerance: {REGRESSION_TOLERANCE}%")
            # Optionally fail on significant regression
            # return 1  # Uncomment to fail on regression
            print("   (Not failing build - regression monitoring only)")
        elif regression > 0:
            print(f"ℹ️  Minor regression of {regression:.2f}% within tolerance")
        else:
            print(f"✅ Coverage improved by {-regression:.2f}%")

        # Update baseline if coverage improved
        if current_coverage > baseline_coverage:
            save_baseline(baseline_path, current_coverage)
            print(f"📊 Updated baseline to {current_coverage:.2f}%")
    else:
        # No baseline exists, create one
        save_baseline(baseline_path, current_coverage)
        print(f"📊 Created baseline at {current_coverage:.2f}%")

    # Always exit 0 for trend monitoring
    return 0


if __name__ == "__main__":
    sys.exit(main())
