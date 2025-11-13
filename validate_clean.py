"""
Validation script to ensure no tool references remain in codebase.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

import os
import re

FORBIDDEN_TERMS = ['claude', 'anthropic', 'ai-generated', 'claude-code']
REQUIRED_AUTHOR = 'Vithushan Jeyapahan'
REQUIRED_EMAIL = 'finance@vijeth.com'


def scan_files(directory):
    """Scan files for forbidden terms and missing attribution."""
    issues = []
    missing_attribution = []

    for root, dirs, files in os.walk(directory):
        # Skip common directories
        if any(d in root for d in ['.git', 'venv', '__pycache__', 'node_modules', '.pytest_cache']):
            continue

        for file in files:
            # Skip the validation script itself
            if file == 'validate_clean.py':
                continue

            if file.endswith(('.py', '.md', '.txt', '.toml', '.yaml', '.yml', '.rst')):
                filepath = os.path.join(root, file)

                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    content_lower = content.lower()

                    # Check for forbidden terms
                    for term in FORBIDDEN_TERMS:
                        if term in content_lower:
                            issues.append(f"{filepath}: contains '{term}'")

                    # Check for proper attribution in Python files
                    if file.endswith('.py') and '__init__' not in file:
                        if REQUIRED_AUTHOR not in content and REQUIRED_EMAIL not in content:
                            missing_attribution.append(filepath)

    return issues, missing_attribution


if __name__ == '__main__':
    issues, missing_attr = scan_files('.')

    print("=" * 60)
    print("CODEBASE VALIDATION REPORT")
    print("=" * 60)

    if issues:
        print("\n❌ Found tool references to remove:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Clean! No tool references found.")

    if missing_attr:
        print(f"\n⚠️  Files missing author attribution ({REQUIRED_AUTHOR}):")
        for filepath in missing_attr:
            print(f"  - {filepath}")
    else:
        print(f"\n✅ All files properly attributed to {REQUIRED_AUTHOR}")

    print("\n" + "=" * 60)

    # Exit with error code if issues found
    if issues or missing_attr:
        exit(1)
    else:
        exit(0)
