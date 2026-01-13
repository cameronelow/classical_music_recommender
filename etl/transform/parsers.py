"""
Parsers for extracting structured information from text fields.

This module provides utilities for parsing:
- Catalog numbers (BWV, Op., K., etc.)
- Work types (Symphony, Concerto, etc.)
- Musical keys (C major, F# minor, etc.)
- Classical periods from dates
"""

import re
from typing import Optional, Dict, List, Tuple
from datetime import datetime


class CatalogNumberParser:
    """Extract catalog numbers from work titles."""

    # Common catalog number patterns
    PATTERNS = [
        # BWV (Bach-Werke-Verzeichnis)
        (r'\bBWV\s*(\d+[a-z]?)', 'BWV'),
        (r'\bS\.\s*(\d+)', 'BWV'),  # Alternative BWV notation

        # Opus numbers
        (r'\bOp(?:us)?\.?\s*(\d+(?:\s*[Nn]o\.?\s*\d+)?[a-z]?)', 'Op.'),

        # Köchel (Mozart)
        (r'\bK\.?\s*(\d+[a-z]?)', 'K.'),
        (r'\bKV\.?\s*(\d+[a-z]?)', 'K.'),

        # Hoboken (Haydn)
        (r'\bH(?:ob)?\.?\s*([IVXLCDM]+[:/]\d+)', 'Hob.'),

        # Deutsch (Schubert)
        (r'\bD\.?\s*(\d+[a-z]?)', 'D.'),

        # Ryom (Vivaldi)
        (r'\bRV\.?\s*(\d+[a-z]?)', 'RV'),

        # Dvořák catalog
        (r'\bB\.?\s*(\d+[a-z]?)', 'B.'),

        # Wq (C.P.E. Bach)
        (r'\bWq\.?\s*(\d+)', 'Wq.'),

        # Generic catalog numbers
        (r'\bNo\.?\s*(\d+)', 'No.'),
    ]

    @classmethod
    def parse(cls, title: str) -> Optional[str]:
        """
        Extract catalog number from title.

        Args:
            title: Work title

        Returns:
            Catalog number string (e.g., 'BWV 1048') or None
        """
        if not title:
            return None

        for pattern, prefix in cls.PATTERNS:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                number = match.group(1).strip()
                return f"{prefix} {number}"

        return None

    @classmethod
    def parse_all(cls, title: str) -> List[str]:
        """
        Extract all catalog numbers from title.

        Args:
            title: Work title

        Returns:
            List of catalog number strings
        """
        if not title:
            return []

        results = []
        for pattern, prefix in cls.PATTERNS:
            for match in re.finditer(pattern, title, re.IGNORECASE):
                number = match.group(1).strip()
                results.append(f"{prefix} {number}")

        return results


class WorkTypeParser:
    """Extract work type from title."""

    # Well-known work titles in foreign languages
    # Check these first before falling back to generic patterns
    KNOWN_TITLES = [
        # The Four Seasons (Vivaldi)
        (r'quattro stagioni', 'Concerto'),  # Italian
        (r'vier jahreszeiten', 'Concerto'),  # German
        (r'cuatro estaciones', 'Concerto'),  # Spanish
        (r'quatres? saisons', 'Concerto'),  # French (including common misspelling "quatres")
        (r'four seasons', 'Concerto'),  # English

        # The Seasons (Haydn)
        (r'jahreszeiten', 'Oratorio'),  # German

        # Messiah
        (r'messias', 'Oratorio'),  # German
        (r'mesías', 'Oratorio'),  # Spanish
        (r'messie', 'Oratorio'),  # French

        # The Magic Flute
        (r'zauberflöte', 'Opera'),  # German
        (r'flauta mágica', 'Opera'),  # Spanish
        (r'flûte enchantée', 'Opera'),  # French

        # Brandenburg Concertos
        (r'brandenburgische konzerte', 'Concerto'),  # German
        (r'conciertos de brandemburgo', 'Concerto'),  # Spanish

        # Water Music
        (r'wassermusik', 'Suite'),  # German
        (r'música acuática', 'Suite'),  # Spanish
        (r'musique pour les feux', 'Suite'),  # French

        # Goldberg Variations
        (r'goldberg.?variationen', 'Variation'),  # German
        (r'variaciones goldberg', 'Variation'),  # Spanish

        # Specific classical works
        (r'cimento dell.?armonia', 'Concerto'),  # Vivaldi's Op. 8
        (r'la mer(?!\s+/)', 'Tone Poem'),  # Debussy (but not "La Mer / Something")
        (r'iberia', 'Suite'),  # Debussy or Albéniz
        (r'préludes', 'Prelude'),  # French
        (r'nocturnes', 'Nocturne'),  # Plural
        (r'études', 'Etude'),  # French plural
        (r'images(?!\s+/)', 'Suite'),  # Debussy (but not compilations)
    ]

    # Work type patterns (order matters - more specific first)
    WORK_TYPES = [
        # Orchestral
        ('Symphony', [r'\bSymphon(?:y|ie)\b', r'\bSinfoni[ae]\b']),
        ('Concerto', [
            r'\bConcert(?:o|i|os)\b',
            r'\bKonzert[e]?\b',  # German
            r'\bConcert[s]?\b(?!\s+pour)',  # English/French (but not "Concert pour" alone)
        ]),
        ('Overture', [r'\bOverture\b', r'\bOuvertüre\b', r'\bObertura\b']),
        ('Suite', [r'\bSuite\b']),
        ('Serenade', [r'\bSerenade\b', r'\bSerenata\b']),
        ('Divertimento', [r'\bDivertimento\b']),
        ('Tone Poem', [r'\bTone Poem\b', r'\bSymphonic Poem\b', r'\bPoema Sinfónico\b']),

        # Chamber
        ('String Quartet', [r'\bString Quartet\b', r'\bStreichquartett\b', r'\bQuatuor à cordes\b', r'\bCuarteto de cuerdas\b']),
        ('Piano Trio', [r'\bPiano Trio\b', r'\bTrio (?:pour|with) Piano\b']),
        ('String Quintet', [r'\bString Quintet\b', r'\bQuintett\b']),
        ('Piano Quartet', [r'\bPiano Quartet\b']),
        ('Quartet', [r'\bQuartet\b', r'\bQuartett\b', r'\bQuatuor\b', r'\bCuarteto\b']),
        ('Trio', [r'\bTrio\b']),

        # Keyboard
        ('Sonata', [r'\bSonat[ae]\b']),
        ('Prelude', [r'\bPr[eé]ludes?\b', r'\bPräludium\b', r'\bPreludio\b']),
        ('Fugue', [r'\bFugue\b', r'\bFuga\b']),
        ('Toccata', [r'\bToccata\b']),
        ('Variation', [r'\bVariations?\b', r'\bVariationen\b', r'\bVariaciones\b']),
        ('Etude', [r'\b[EÉ]tudes?\b', r'\bStudy\b', r'\bEstudio\b']),
        ('Nocturne', [r'\bNocturnes?\b', r'\bNotturno\b']),
        ('Waltz', [r'\bWaltz\b', r'\bValse\b', r'\bWalzer\b']),
        ('Mazurka', [r'\bMazurka\b']),
        ('Polonaise', [r'\bPolonaise\b', r'\bPolonesa\b']),
        ('Impromptu', [r'\bImpromptu\b']),
        ('Fantasia', [r'\bFantasi[ae]\b', r'\bFantasy\b', r'\bFantasía\b']),
        ('Ballade', [r'\bBallade\b', r'\bBalada\b']),
        ('Intermezzo', [r'\bIntermezzo\b']),

        # Vocal - keep specific ones first
        ('Mass', [r'\bMass\b', r'\bMess[ae]\b', r'\bMissa\b']),
        ('Requiem', [r'\bRequiem\b', r'\bRéquiem\b']),
        ('Cantata', [r'\bCantata\b', r'\bKantate\b']),
        ('Oratorio', [r'\bOratorio\b']),
        ('Opera', [r'\bOpera\b', r'\bÓpera\b', r'\bOper\b']),
        ('Motet', [r'\bMotet\b', r'\bMotetto\b']),
        ('Passion', [r'\bPassion\b']),
        ('Magnificat', [r'\bMagnificat\b']),
        ('Gloria', [r'\bGloria\b']),
        ('Stabat Mater', [r'\bStabat Mater\b']),
        ('Te Deum', [r'\bTe Deum\b']),
        ('Song', [r'\bLied(?:er)?\b', r'\bSong\b', r'\bCanción\b', r'\bChanson\b']),
        ('Hymn', [r'\bHymn\b', r'\bHimno\b']),
        ('Psalm', [r'\bPsalm\b', r'\bSalmo\b', r'\bPsaume\b']),
        ('Choral', [r'\bChoral\b', r'\bChorale\b']),

        # Other
        ('Concertino', [r'\bConcertino\b']),
        ('Capriccio', [r'\bCapriccio\b']),
        ('Rhapsody', [r'\bRhapsod(?:y|ie)\b', r'\bRapsodia\b']),
        ('Scherzo', [r'\bScherzo\b']),
        ('March', [r'\bMarch\b', r'\bMarsch\b', r'\bMarcha\b']),
        ('Dance', [r'\bDance\b', r'\bTanz\b', r'\bDanza\b']),
        ('Rondo', [r'\bRondo\b', r'\bRondó\b']),
        ('Movement', [r'\bMovement\b', r'\bSatz\b', r'\bMovimiento\b']),
    ]

    @classmethod
    def parse(cls, title: str) -> Optional[str]:
        """
        Extract work type from title.

        Args:
            title: Work title

        Returns:
            Work type string or None
        """
        if not title:
            return None

        # First, check for known work titles in foreign languages
        for pattern, work_type in cls.KNOWN_TITLES:
            if re.search(pattern, title, re.IGNORECASE):
                return work_type

        # Then fall back to generic work type patterns
        for work_type, patterns in cls.WORK_TYPES:
            for pattern in patterns:
                if re.search(pattern, title, re.IGNORECASE):
                    return work_type

        return None

    @classmethod
    def parse_all(cls, title: str) -> List[str]:
        """
        Extract all work types from title.

        Args:
            title: Work title

        Returns:
            List of work type strings
        """
        if not title:
            return []

        results = []
        for work_type, patterns in cls.WORK_TYPES:
            for pattern in patterns:
                if re.search(pattern, title, re.IGNORECASE):
                    results.append(work_type)
                    break

        return results


class KeyParser:
    """Extract musical key from title."""

    # Note names and accidentals
    NOTES = r'[A-G]'
    ACCIDENTALS = r'(?:[#♯]|[b♭]|-sharp|-flat)?'
    MODES = r'(?:major|minor|maj|min|dur|moll|M|m)'

    # Key patterns (order matters - try more specific first)
    KEY_PATTERNS = [
        # German format: C-Dur, c-moll, A-Moll (with hyphen)
        (rf'\b([A-Ga-g])-([Dd]ur|[Mm]oll)\b', 'german_hyphen'),

        # Standard format: C major, D minor, etc.
        (rf'\b({NOTES}{ACCIDENTALS})\s*({MODES})\b', 'standard'),

        # Short format: "in C" (assume major for uppercase, minor for lowercase)
        # Only match if followed by comma, space+word, or end of string to avoid false positives
        (r'\bin\s+([A-G](?:[#♯b♭])?)\s*(?:,|\s+[A-Z]|$)', 'short_major'),
        (r'\bin\s+([a-g](?:[#♯b♭])?)\s*(?:,|\s+[A-Z]|$)', 'short_minor'),
    ]

    # Normalize accidentals
    SHARP_SYMBOLS = ['#', '♯', '-sharp']
    FLAT_SYMBOLS = ['b', '♭', '-flat']

    @classmethod
    def parse(cls, title: str) -> Optional[str]:
        """
        Extract musical key from title.

        Args:
            title: Work title

        Returns:
            Key string (e.g., 'D minor', 'E♭ major') or None
        """
        if not title:
            return None

        # Try each pattern in order
        for pattern, pattern_type in cls.KEY_PATTERNS:
            match = re.search(pattern, title, re.IGNORECASE if pattern_type == 'standard' else 0)
            if match:
                return cls._parse_match(match, pattern_type)

        return None

    @classmethod
    def _parse_match(cls, match: re.Match, pattern_type: str) -> str:
        """Parse a regex match based on pattern type."""
        if pattern_type == 'german_hyphen':
            # German format: C-Dur, c-moll
            note = match.group(1)
            mode_german = match.group(2).lower()

            # Normalize note (uppercase first letter)
            note = cls._normalize_note(note.upper())

            # Convert German mode to English
            mode = 'major' if mode_german == 'dur' else 'minor'

            return f"{note} {mode}"

        elif pattern_type == 'short_major':
            # Short format "in C" - assume major
            note = match.group(1)
            note = cls._normalize_note(note)
            return f"{note} major"

        elif pattern_type == 'short_minor':
            # Short format "in c" - assume minor
            note = match.group(1)
            note = cls._normalize_note(note.upper())
            return f"{note} minor"

        else:  # standard
            # Standard format: C major, D minor
            note = match.group(1)
            mode = match.group(2).lower()

            # Normalize note
            note = cls._normalize_note(note)

            # Normalize mode
            if mode in ['maj', 'dur']:
                mode = 'major'
            elif mode in ['min', 'moll']:
                mode = 'minor'
            elif mode == 'm':
                # Ambiguous - could be major or minor, default to major
                mode = 'major'

            return f"{note} {mode}"

    @classmethod
    def _normalize_note(cls, note: str) -> str:
        """Normalize note representation."""
        # Keep first letter uppercase
        base = note[0].upper()
        rest = note[1:].lower()

        # Normalize sharps
        for sharp in cls.SHARP_SYMBOLS:
            if sharp in rest:
                return f"{base}♯"

        # Normalize flats
        for flat in cls.FLAT_SYMBOLS:
            if flat in rest:
                return f"{base}♭"

        return base


class PeriodClassifier:
    """Classify composer into historical period based on dates."""

    PERIODS = [
        ('Medieval', None, 1400),
        ('Renaissance', 1400, 1600),
        ('Baroque', 1600, 1750),
        ('Classical', 1750, 1820),
        ('Romantic', 1820, 1910),
        ('Modern', 1910, 2000),
        ('Contemporary', 2000, None),
    ]

    @classmethod
    def classify(cls, birth_year: Optional[int], death_year: Optional[int]) -> Optional[str]:
        """
        Classify composer period based on birth/death years.

        Uses the midpoint of the composer's life or their most productive years.

        Args:
            birth_year: Year of birth
            death_year: Year of death

        Returns:
            Period name or None
        """
        if not birth_year:
            return None

        # Use midpoint of life if we have both dates
        if death_year:
            reference_year = birth_year + (death_year - birth_year) // 2
        else:
            # Assume peak productivity around age 40
            reference_year = birth_year + 40

        for period, start, end in cls.PERIODS:
            if start is None and reference_year < end:
                return period
            elif end is None and reference_year >= start:
                return period
            elif start and end and start <= reference_year < end:
                return period

        return None

    @classmethod
    def classify_from_dates(cls, birth_date: Optional[str], death_date: Optional[str]) -> Optional[str]:
        """
        Classify period from date strings.

        Args:
            birth_date: Birth date string (ISO format)
            death_date: Death date string (ISO format)

        Returns:
            Period name or None
        """
        birth_year = cls._extract_year(birth_date)
        death_year = cls._extract_year(death_date)

        return cls.classify(birth_year, death_year)

    @staticmethod
    def _extract_year(date_str: Optional[str]) -> Optional[int]:
        """Extract year from ISO date string."""
        if not date_str:
            return None

        try:
            # Try ISO format
            if '-' in date_str:
                return int(date_str.split('-')[0])
            # Try just year
            return int(date_str)
        except (ValueError, IndexError):
            return None
