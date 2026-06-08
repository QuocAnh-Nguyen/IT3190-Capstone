"""Genre consolidation: map fine-grained genres to broader parent categories.

This is critical for model performance — the original 992 unique genre labels
from Spotify are far too granular for a 10K-sample dataset. We consolidate
sub-genres into ~10-15 broader, more learnable categories.
"""

from typing import Dict, List

# Maps sub-genres to parent categories.
# Only genres present in the dataset after Phase 1 filtering are included.
GENRE_MAPPING: Dict[str, str] = {
    # --- Rock & Alternative ---
    "album rock": "Rock",
    "alternative rock": "Rock",
    "classic rock": "Rock",
    "art rock": "Rock",
    "blues-rock": "Rock",
    "glam rock": "Rock",
    "dance rock": "Rock",
    "alternative metal": "Metal",
    "glam metal": "Metal",
    "neo mellow": "Rock",

    # --- Hip Hop / Rap ---
    "atl hip hop": "Hip Hop",
    "east coast hip hop": "Hip Hop",
    "conscious hip hop": "Hip Hop",
    "canadian hip hop": "Hip Hop",
    "chicago rap": "Hip Hop",
    "alternative hip hop": "Hip Hop",
    "hip hop": "Hip Hop",

    # --- Pop ---
    "dance pop": "Pop",
    "pop": "Pop",
    "bubblegum pop": "Pop",
    "boy band": "Pop",
    "canadian pop": "Pop",
    "australian pop": "Pop",
    "new wave pop": "Pop",
    "brill building pop": "Pop",

    # --- R&B / Soul ---
    "classic soul": "R&B / Soul",
    "funk": "R&B / Soul",
    "motown": "R&B / Soul",
    "disco": "R&B / Soul",
    "freestyle": "R&B / Soul",

    # --- Country & Folk ---
    "contemporary country": "Country & Folk",
    "country": "Country & Folk",
    "folk": "Country & Folk",

    # --- Traditional / Standards ---
    "adult standards": "Traditional / Standards",

    # --- Other (keep as-is) ---
    "Other": "Other",
}


def get_consolidated_genre_map() -> Dict[str, str]:
    """Return the genre consolidation mapping."""
    return GENRE_MAPPING.copy()


def get_parent_genres() -> List[str]:
    """Return sorted list of unique parent genre categories."""
    return sorted(set(GENRE_MAPPING.values()))


def apply_genre_consolidation(series):
    """Apply the genre mapping to a pandas Series of genre labels.

    Any genre not in the mapping remains unchanged.
    """
    return series.apply(lambda g: GENRE_MAPPING.get(g, g))