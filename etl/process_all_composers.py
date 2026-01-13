"""
Process all composers through the complete ETL pipeline.

This script:
1. Extracts data from MusicBrainz for all specified composers
2. Transforms the raw data into normalized parquet tables
3. Converts parquet files to CSV
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from etl.extract import BatchProcessor, Config
from etl.transform import MusicDataTransformer
import subprocess


def main():
    """Run the complete ETL pipeline."""
    print("=" * 80)
    print("CLASSICAL MUSIC ETL PIPELINE")
    print("=" * 80)

    # List of composers to process
    # Organized by historical period for better dataset coverage
    composers = [
        # BAROQUE ERA (1600-1750)
        "Johann Sebastian Bach",
        "Antonio Vivaldi",
        "George Frideric Handel",
        "Domenico Scarlatti",
        "Arcangelo Corelli",
        "Henry Purcell",
        "Jean-Philippe Rameau",
        "Tomaso Albinoni",
        "Georg Philipp Telemann",
        "François Couperin",
        "Johann Pachelbel",

        # CLASSICAL ERA (1750-1820)
        "Wolfgang Amadeus Mozart",
        "Ludwig van Beethoven",
        "Joseph Haydn",
        "Carl Philipp Emanuel Bach",
        "Christoph Willibald Gluck",
        "Luigi Boccherini",
        "Muzio Clementi",
        "Antonio Salieri",
        "Domenico Cimarosa",

        # ROMANTIC ERA (1820-1910)
        "Frédéric Chopin",
        "Johannes Brahms",
        "Franz Schubert",
        "Robert Schumann",
        "Franz Liszt",
        "Richard Wagner",
        "Giuseppe Verdi",
        "Antonín Dvořák",
        "Pyotr Ilyich Tchaikovsky",
        "Felix Mendelssohn",
        "Hector Berlioz",
        "Camille Saint-Saëns",
        "Edvard Grieg",
        "Sergei Rachmaninoff",
        "Gustav Mahler",
        "Nikolai Rimsky-Korsakov",
        "César Franck",
        "Gabriel Fauré",
        "Alexander Scriabin",
        "Max Bruch",
        "Bedřich Smetana",
        "Anton Bruckner",
        "Giacomo Puccini",
        "Richard Strauss",
        "Jean Sibelius",
        "Edward Elgar",

        # IMPRESSIONIST/EARLY MODERN (1880-1920)
        "Claude Debussy",
        "Maurice Ravel",
        "Erik Satie",

        # MODERN ERA (1910-2000)
        "Igor Stravinsky",
        "Sergei Prokofiev",
        "Dmitri Shostakovich",
        "Béla Bartók",
        "Arnold Schoenberg",
        "Benjamin Britten",
        "Aaron Copland",
        "George Gershwin",
        "Samuel Barber",
        "Leonard Bernstein",
        "Olivier Messiaen",
        "Paul Hindemith",
        "Alban Berg",
        "Anton Webern",
        "Kurt Weill",
        "Ralph Vaughan Williams",
        "Heitor Villa-Lobos",
        "Manuel de Falla",
        "Carl Orff",
        "Aram Khachaturian"
    ]

    # STEP 1: Extract from MusicBrainz
    print("\n" + "=" * 80)
    print("STEP 1: EXTRACTING DATA FROM MUSICBRAINZ")
    print("=" * 80)

    config = Config()

    try:
        config.validate()
        print("\n✓ Configuration validated")
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        print("\nPlease create a .env file with MUSICBRAINZ_EMAIL set.")
        return 1

    processor = BatchProcessor(config)

    print(f"\nProcessing {len(composers)} composers...")
    stats = processor.process_artists(composers, resume=True)

    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total composers: {stats['total']}")
    print(f"Already completed: {stats['already_completed']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")

    if stats['errors']:
        print("\nErrors encountered:")
        for error in stats['errors']:
            print(f"  - {error['artist']}: {error['error']}")

    # STEP 2: Transform to normalized tables
    print("\n" + "=" * 80)
    print("STEP 2: TRANSFORMING DATA")
    print("=" * 80)

    transformer = MusicDataTransformer(
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR
    )

    try:
        metrics = transformer.transform_all()
        metrics.print_report()
    except Exception as e:
        print(f"\n✗ Transformation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # STEP 3: Convert to CSV
    print("\n" + "=" * 80)
    print("STEP 3: CONVERTING TO CSV")
    print("=" * 80)

    try:
        result = subprocess.run(
            ["python3", "etl/convert_parquet_to_csv.py"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ CSV conversion failed: {e}")
        print(e.stderr)
        return 1

    # Final summary
    print("\n" + "=" * 80)
    print("ETL PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nOutput files:")
    print(f"  Raw data: {config.RAW_DATA_DIR}")
    print(f"  Processed data: {config.PROCESSED_DATA_DIR}")
    print("\nProcessed tables:")
    print("  - composers.parquet / composers.csv")
    print("  - works.parquet / works.csv")
    print("  - work_tags.parquet / work_tags.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
