"""Shared review normalisation used by training and Flask inference."""

import re
import unicodedata


def normalize_review(text: str) -> str:
    if text is None:
        return ""
    folded = unicodedata.normalize("NFKC", str(text)).lower()
    folded = re.sub(r"[^a-z\s]", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()
