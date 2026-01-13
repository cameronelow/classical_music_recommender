#!/usr/bin/env python3
"""
Fix data quality issues in the classical music recommender dataset.

Issues addressed:
1. Works with UUID as title (corrupt data)
2. Works with missing titles
3. Works with missing composers
4. Duplicate works
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import re
from recommender.config import get_config

def find_data_issues():
    """Find all data quality issues."""

    config = get_config()

    # Load data
    works_path = config.paths.works_parquet
    if not works_path.exists():
        print(f"Error: Works file not found at {works_path}")
        return

    works_df = pd.read_parquet(works_path)

    print("=" * 80)
    print("DATA QUALITY ANALYSIS")
    print("=" * 80)
    print(f"\nTotal works: {len(works_df)}")

    issues = []

    # Issue 1: UUID as title
    uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
    uuid_titles = works_df[works_df['title'].str.match(uuid_pattern, case=False, na=False)]

    if len(uuid_titles) > 0:
        print(f"\n⚠️  {len(uuid_titles)} works with UUID as title:")
        for _, work in uuid_titles.iterrows():
            print(f"  - Work ID: {work['work_id']}")
            print(f"    Title: {work['title']}")
            print(f"    Composer ID: {work.get('composer_id', 'N/A')}")
            issues.append({
                'type': 'uuid_title',
                'work_id': work['work_id'],
                'title': work['title']
            })

    # Issue 2: Missing or empty titles
    missing_titles = works_df[works_df['title'].isna() | (works_df['title'] == '')]
    if len(missing_titles) > 0:
        print(f"\n⚠️  {len(missing_titles)} works with missing title:")
        for _, work in missing_titles.head(10).iterrows():
            print(f"  - Work ID: {work['work_id']}")
            issues.append({
                'type': 'missing_title',
                'work_id': work['work_id']
            })

    # Issue 3: Works with 'Unknown' or generic titles
    generic_titles = works_df[works_df['title'].str.contains('Unknown|Untitled|Unnamed', case=False, na=False)]
    if len(generic_titles) > 0:
        print(f"\n⚠️  {len(generic_titles)} works with generic title:")
        for _, work in generic_titles.head(5).iterrows():
            print(f"  - {work['title']} (ID: {work['work_id']})")

    # Issue 4: Duplicate titles
    duplicate_titles = works_df[works_df.duplicated('title', keep=False)]['title'].value_counts()
    if len(duplicate_titles) > 0:
        print(f"\n⚠️  {len(duplicate_titles)} duplicate titles found")
        print(f"   (These may be legitimate - same work by different composers)")
        for title, count in duplicate_titles.head(5).items():
            print(f"  - '{title}' appears {count} times")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"UUID titles: {len(uuid_titles)}")
    print(f"Missing titles: {len(missing_titles)}")
    print(f"Generic titles: {len(generic_titles)}")
    print(f"Duplicate titles: {len(duplicate_titles)}")

    return issues, works_df


def fix_uuid_title(work_id: str, works_df: pd.DataFrame) -> str:
    """Attempt to fix a UUID title by looking at related data."""

    work = works_df[works_df['work_id'] == work_id].iloc[0]

    # Option 1: Use catalog number if available
    if pd.notna(work.get('catalog_number')):
        catalog = work['catalog_number']
        # Look for other works with same catalog
        similar = works_df[
            (works_df['catalog_number'] == catalog) &
            (works_df['work_id'] != work_id)
        ]
        if len(similar) > 0 and not similar.iloc[0]['title'].startswith(work['title'][:8]):
            base_title = similar.iloc[0]['title']
            return f"{base_title} (alternate)"

    # Option 2: Use work_type + catalog number
    if pd.notna(work.get('work_type')) and pd.notna(work.get('catalog_number')):
        return f"{work['work_type']} {work['catalog_number']}"

    # Option 3: Use work_type + key
    if pd.notna(work.get('work_type')) and pd.notna(work.get('key')):
        return f"{work['work_type']} in {work['key']}"

    # Fallback: Mark as unknown
    return f"Unknown Work ({work.get('catalog_number', 'no catalog')})"


def apply_fixes(issues, works_df):
    """Apply fixes to data issues."""

    print("\n" + "=" * 80)
    print("APPLYING FIXES")
    print("=" * 80)

    fixed_count = 0

    for issue in issues:
        if issue['type'] == 'uuid_title':
            work_id = issue['work_id']
            old_title = issue['title']

            new_title = fix_uuid_title(work_id, works_df)

            # Update the dataframe
            works_df.loc[works_df['work_id'] == work_id, 'title'] = new_title

            print(f"\n✅ Fixed UUID title:")
            print(f"   Work ID: {work_id}")
            print(f"   Old: {old_title}")
            print(f"   New: {new_title}")

            fixed_count += 1

    print(f"\n{'-'*80}")
    print(f"Fixed {fixed_count} issues")

    return works_df


def save_cleaned_data(works_df):
    """Save the cleaned dataset."""

    config = get_config()
    output_path = config.paths.works_parquet
    backup_path = output_path.parent / f"{output_path.stem}_backup{output_path.suffix}"

    print("\n" + "=" * 80)
    print("SAVING CLEANED DATA")
    print("=" * 80)

    # Create backup
    import shutil
    shutil.copy2(output_path, backup_path)
    print(f"✅ Backup created: {backup_path}")

    # Save cleaned data
    works_df.to_parquet(output_path, index=False)
    print(f"✅ Cleaned data saved: {output_path}")

    print("\n⚠️  IMPORTANT: You must restart the backend for changes to take effect!")
    print("   Also delete embedding cache: rm -rf .cache/embeddings/")


def main():
    """Main function."""

    print("\n" + "=" * 80)
    print("CLASSICAL MUSIC DATA QUALITY FIXER")
    print("=" * 80)

    # Find issues
    issues, works_df = find_data_issues()

    if not issues:
        print("\n✅ No data quality issues found!")
        return

    # Ask for confirmation
    print("\n" + "=" * 80)
    response = input("\nApply fixes? (yes/no): ").strip().lower()

    if response in ['yes', 'y']:
        works_df = apply_fixes(issues, works_df)
        save_cleaned_data(works_df)

        print("\n" + "=" * 80)
        print("✅ DATA CLEANING COMPLETE!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Restart backend server")
        print("  2. Delete embedding cache: rm -rf .cache/embeddings/")
        print("  3. Test searches - corrupt work should now have proper title")
    else:
        print("\nNo changes made.")


if __name__ == "__main__":
    main()
