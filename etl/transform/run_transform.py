"""
Run the data transformation pipeline.

This script transforms raw extracted data into normalized Parquet tables.

Usage:
    python run_transform.py
    python run_transform.py --input-dir ../data/raw --output-dir ../data/processed
    python run_transform.py --artists Bach Dvorak
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from etl.transform import MusicDataTransformer


def main():
    """Run transformation pipeline."""
    parser = argparse.ArgumentParser(
        description='Transform raw music data into normalized Parquet tables'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=project_root / 'data' / 'raw',
        help='Input directory containing raw data files (default: data/raw)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=project_root / 'data' / 'processed',
        help='Output directory for Parquet files (default: data/processed)'
    )
    parser.add_argument(
        '--artists',
        nargs='+',
        default=None,
        help='Specific artists to process (default: auto-detect from files)'
    )
    parser.add_argument(
        '--report',
        type=Path,
        default=None,
        help='Path to save quality report JSON (optional)'
    )

    args = parser.parse_args()

    # Print header
    print("=" * 80)
    print("CLASSICAL MUSIC DATA TRANSFORMATION")
    print("=" * 80)
    print(f"\nInput directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.artists:
        print(f"Artists:          {', '.join(args.artists)}")
    else:
        print(f"Artists:          Auto-detect")
    print()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"ERROR: Input directory does not exist: {args.input_dir}")
        return 1

    # Create transformer
    transformer = MusicDataTransformer(args.input_dir, args.output_dir)

    # Run transformation
    try:
        metrics = transformer.transform_all(args.artists)
    except Exception as e:
        print(f"\nERROR: Transformation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print quality report
    metrics.print_report()

    # Save report if requested
    if args.report:
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'input_dir': str(args.input_dir),
            'output_dir': str(args.output_dir),
            'artists': args.artists,
            'metrics': metrics.to_dict()
        }

        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)

        print(f"\nQuality report saved to: {args.report}")

    # Print output files
    print("\nOUTPUT FILES:")
    for file in sorted(args.output_dir.glob('*.parquet')):
        size_kb = file.stat().st_size / 1024
        print(f"  {file.name:30s} {size_kb:8.2f} KB")

    print("\n" + "=" * 80)
    print("TRANSFORMATION COMPLETE")
    print("=" * 80)

    # Return error code if there were errors
    if metrics.errors:
        print(f"\nWARNING: Completed with {len(metrics.errors)} errors")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
