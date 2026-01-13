"""Convert work_tags_enhanced.csv to parquet format for use in recommender."""
import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'processed'

    csv_path = data_dir / 'work_tags_enhanced.csv'
    parquet_path = data_dir / 'work_tags_enhanced.parquet'

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    # Load CSV
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} tag associations")
    print(f"  - {df['work_id'].nunique()} unique works")
    print(f"  - {df['tag'].nunique()} unique tags")

    # Show source breakdown
    if 'source' in df.columns:
        print("\nTag source breakdown:")
        print(df['source'].value_counts())

    # Convert to parquet
    print(f"\nSaving to {parquet_path}...")
    df.to_parquet(parquet_path, index=False)

    # Verify
    df_verify = pd.read_parquet(parquet_path)
    print(f"✓ Verified: {len(df_verify)} rows in parquet file")
    print(f"✓ File size: {parquet_path.stat().st_size / 1024:.1f} KB")

    print("\nConversion complete!")

if __name__ == "__main__":
    main()
