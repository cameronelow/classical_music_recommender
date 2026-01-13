import pandas as pd
import os
from pathlib import Path

def convert_parquet_to_csv(parquet_file_path, csv_file_path):
    """Convert a parquet file to CSV format."""
    df = pd.read_parquet(parquet_file_path)
    df.to_csv(csv_file_path, index=False)
    print(f"Converted: {parquet_file_path} -> {csv_file_path}")

def main():
    # Base directory
    base_dir = Path(__file__).parent.parent

    # Directories containing parquet files
    data_dirs = [
        base_dir / 'data' / 'raw',
        base_dir / 'data' / 'processed'
    ]

    for data_dir in data_dirs:
        if data_dir.exists():
            parquet_files = list(data_dir.glob('*.parquet'))

            for parquet_file in parquet_files:
                # Create CSV filename by replacing .parquet extension with .csv
                csv_file = parquet_file.with_suffix('.csv')

                try:
                    convert_parquet_to_csv(parquet_file, csv_file)
                except Exception as e:
                    print(f"Error converting {parquet_file}: {e}")
        else:
            print(f"Directory not found: {data_dir}")

    print("\nConversion complete!")

if __name__ == "__main__":
    main()
