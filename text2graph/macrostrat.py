from functools import cache
import requests
import re

BASE_URL = "https://macrostrat.org/api"


@cache
def get_all_strat_names() -> list:
    """Get all stratigraphic names from macrostrat API."""

    url = f"{BASE_URL}/defs/strat_names?all"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()["success"]["data"]
    return sorted(list(set([x["strat_name"] for x in data])))


def get_strat_records(strat_name=str) -> list[dict]:
    """Get the records for a given stratigraphic name."""

    response = requests.get(f"{BASE_URL}/defs/strat_names?strat_name={strat_name}")
    response.raise_for_status()
    return response.json()["success"]["data"]


def _find_word_occurrences(text: str, search_word: str) -> list[dict]:
    """Find all occurrences of a word in a given text and get its position of occurrence."""
    matches = [match for match in re.finditer(rf"\b{re.escape(search_word)}\b", text)]
    results = [
        {
            "word": match.group(),
            "start": match.start(),
            "end": match.end(),
            "link": f"{BASE_URL}/defs/strat_names?strat_name={match.group()}",
        }
        for match in matches
    ]
    return results


def find_all_occurrences(text: str, words: list[str]) -> list[dict[str, str | int]]:
    """Find all occurrences of a list of terms in a given text and get its position of occurrence."""
    occurrences = []
    for word in words:
        this_occ = _find_word_occurrences(text=text, search_word=word)
        if this_occ:
            occurrences.extend(this_occ)
    return sorted(occurrences, key=lambda x: x["start"])
