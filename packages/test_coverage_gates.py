#!/usr/bin/env python3
"""Test script to validate coverage gates locally."""

import subprocess
import sys
from pathlib import Path


def run_coverage_gate(gate_name: str, cmd: list[str], expected_min: int) -> tuple[bool, int]:
    """Run a coverage gate and return (passed, coverage_percentage)."""
    print(f"\n=== {gate_name} ===")
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        # Extract coverage percentage from output
        lines = result.stdout.split("\n")
        coverage_line = None
        for line in reversed(lines):
            if "TOTAL" in line and "%" in line:
                coverage_line = line
                break

        if coverage_line:
            coverage_pct = int(coverage_line.split()[-1].rstrip("%"))
            print(f"Coverage: {coverage_pct}%")

            if coverage_pct >= expected_min:
                print(f"✅ Meets minimum threshold ({expected_min}%)")
                return True, coverage_pct
            print(f"⚠️  Below minimum threshold ({expected_min}%)")
            return False, coverage_pct
        print("❌ Could not determine coverage percentage")
        return False, 0

    except subprocess.CalledProcessError as e:
        print(f"❌ {gate_name} FAILED")
        print(f"Error: {e}")
        return False, 0


def main():
    """Test both coverage gates locally (mimics CI behavior)."""
    repo_root = Path(__file__).parent

    print("Testing Coverage Gates Locally")
    print("=" * 50)

    # First run pytest to generate coverage data
    print("\n=== Running tests with coverage ===")
    test_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--cov=glassalpha",
        "--cov-report=xml",
        "--cov-report=term-missing",
        "--cov-fail-under=0",  # No global fail-under
        "tests/",
    ]
    subprocess.run(test_cmd, check=False)

    # Gate 1: Critical Path (90%+ hard requirement)
    print("\n" + "=" * 50)
    gate1_cmd = [
        sys.executable,
        "-m",
        "coverage",
        "report",
        "--precision=2",
        "--include=*/glassalpha/config/*,*/glassalpha/models/*,*/glassalpha/pipeline/*,*/glassalpha/report/renderer.py,*/glassalpha/metrics/core.py,*/glassalpha/metrics/thresholds.py,*/glassalpha/core/interfaces.py",
        "--fail-under=90",
    ]

    gate1_passed, gate1_coverage = run_coverage_gate(
        "Gate 1: Critical-path modules (90%+)",
        gate1_cmd,
        90,
    )

    # Gate 2: Full Repository (70% trend-only, no fail)
    print("\n" + "=" * 50)
    gate2_cmd = [
        sys.executable,
        "-m",
        "coverage",
        "report",
        "--precision=2",
        "--fail-under=0",  # Trend only, no fail
    ]

    gate2_passed, gate2_coverage = run_coverage_gate(
        "Gate 2: Full repository (70% trend-only)",
        gate2_cmd,
        70,
    )

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Gate 1 (Critical Path): {gate1_coverage}% {'✅' if gate1_passed else '❌'}")
    print(f"Gate 2 (Full Repository): {gate2_coverage}% {'✅' if gate2_passed else '⚠️'}")

    if gate1_passed:
        print("✅ Gate 1 (Critical Path) meets 90% threshold")
    else:
        print("❌ Gate 1 (Critical Path) below 90% threshold - THIS IS A HARD FAIL")

    if gate2_passed:
        print("✅ Gate 2 (Full Repository) meets 70% threshold")
    else:
        print("⚠️  Gate 2 (Full Repository) below 70% threshold - trend monitoring only (not a failure)")

    # Only Gate 1 is a hard requirement
    return gate1_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
