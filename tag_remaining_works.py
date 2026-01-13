#!/usr/bin/env python3
"""
DEPRECATED: This script is now a wrapper around manage_tagging.py

For new usage, please use:
    python3 manage_tagging.py tag-all

This file is kept for backward compatibility.
"""

import sys
import subprocess
from pathlib import Path

print("=" * 80)
print("NOTE: tag_remaining_works.py is deprecated")
print("=" * 80)
print("\nThis script now calls the unified tagging system.")
print("For more features and options, use:")
print("  python3 manage_tagging.py --help")
print("\nRunning: python3 manage_tagging.py tag-all")
print("=" * 80)
print()

# Call the new unified script
script_path = Path(__file__).parent / "manage_tagging.py"
result = subprocess.run([sys.executable, str(script_path), "tag-all"], check=False)

sys.exit(result.returncode)
