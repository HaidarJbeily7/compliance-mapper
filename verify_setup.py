#!/usr/bin/env python3
"""
Verify that the compliance-mapper setup is complete and correct.
"""

import sys
from pathlib import Path


def check_file(path: str, description: str) -> bool:
    """Check if a file exists."""
    if Path(path).exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} (MISSING)")
        return False


def check_directory(path: str, description: str) -> bool:
    """Check if a directory exists."""
    if Path(path).is_dir():
        print(f"✓ {description}: {path}/")
        return True
    else:
        print(f"✗ {description}: {path}/ (MISSING)")
        return False


def main():
    """Run verification checks."""
    print("=" * 80)
    print("COMPLIANCE MAPPER - SETUP VERIFICATION")
    print("=" * 80)
    print()
    
    checks = []
    
    # Core files
    print("Core Files:")
    checks.append(check_file("requirements.txt", "Dependencies"))
    checks.append(check_file("config.yaml", "Configuration"))
    checks.append(check_file("run_benchmarks.py", "Benchmark runner"))
    checks.append(check_file("generate_report.py", "Report generator"))
    print()
    
    # Benchmark modules
    print("Benchmark Modules:")
    checks.append(check_file("benchmarks/__init__.py", "Benchmarks package"))
    checks.append(check_file("benchmarks/base.py", "Base classes"))
    checks.append(check_file("benchmarks/model_client.py", "Model client"))
    checks.append(check_file("benchmarks/trust_eval_runner.py", "TrustEval adapter"))
    checks.append(check_file("benchmarks/decoding_trust_runner.py", "DecodingTrust adapter"))
    print()
    
    # Compliance modules
    print("Compliance Modules:")
    checks.append(check_file("compliance/__init__.py", "Compliance package"))
    checks.append(check_file("compliance/mapper.py", "Compliance mapper"))
    checks.append(check_file("compliance/report_builder.py", "Report builder"))
    checks.append(check_file("compliance/eu_ai_act_matrix.csv", "EU AI Act matrix"))
    print()
    
    # Output directories
    print("Output Directories:")
    checks.append(check_directory("metrics", "Metrics directory"))
    checks.append(check_directory("reports", "Reports directory"))
    print()
    
    # Documentation
    print("Documentation:")
    checks.append(check_file("README.md", "Project README"))
    checks.append(check_file("SETUP.md", "Setup guide"))
    checks.append(check_file("QUICKSTART.md", "Quick start guide"))
    checks.append(check_file("FINDINGS.md", "Findings & analysis"))
    checks.append(check_file("PROJECT_SUMMARY.md", "Project summary"))
    print()
    
    # Optional files
    print("Optional Files:")
    env_exists = Path(".env").exists()
    if env_exists:
        print(f"✓ Environment file: .env")
    else:
        print(f"⚠ Environment file: .env (not found - copy from .env.example)")
    print()
    
    # Summary
    print("=" * 80)
    total = len(checks)
    passed = sum(checks)
    
    if passed == total:
        print(f"✓ ALL CHECKS PASSED ({passed}/{total})")
        print()
        print("Setup is complete! Next steps:")
        print("  1. Copy .env.example to .env and add your API keys")
        print("  2. Run: python run_benchmarks.py")
        print("  3. Run: python generate_report.py")
        print()
        print("See QUICKSTART.md for detailed instructions.")
        return 0
    else:
        print(f"✗ SOME CHECKS FAILED ({passed}/{total} passed)")
        print()
        print("Please ensure all required files are present.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

