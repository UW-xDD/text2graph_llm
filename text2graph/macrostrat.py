from functools import cache
import requests
import re

BASE_URL = "https://macrostrat.org/api"


@cache
def get_all_strat_names() -> list[str]:
    """Get all stratigraphic names from macrostrat API."""

    url = f"{BASE_URL}/defs/strat_names?all"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()["success"]["data"]
    return sorted(list(set([x["strat_name"] for x in data])))


def get_all_intervals() -> list[dict]:
    """Get all intervals from macrostrat API."""

    url = f"{BASE_URL}/defs/intervals?all"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()["success"]["data"]
    return data


@cache
def get_all_lithologies() -> list[str]:
    """Get all lithologies from macrostrat API."""

    url = f"{BASE_URL}/defs/lithologies?all"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()["success"]["data"]
    return sorted(list(set([x["name"] for x in data])))


def get_known_entities() -> dict:
    """Get known entities for annotations."""
    lithologies = get_all_lithologies()
    strat_names = get_all_strat_names()
    return {
        **{lith: "lithology" for lith in lithologies},
        **{strat: "strat name" for strat in strat_names},
    }


@cache
def get_strat_records(strat_name=str, exact: bool = False) -> list[dict]:
    """Get the records for a given stratigraphic name."""

    response = requests.get(f"{BASE_URL}/defs/strat_names?strat_name={strat_name}")
    response.raise_for_status()
    matches = response.json()["success"]["data"]
    if exact:
        matches = [match for match in matches if match["strat_name"] == strat_name]
    if not matches:
        raise ValueError(f"No stratigraphic name found for '{strat_name}'")
    return matches


@cache
def get_lith_records(lith_name=str, exact: bool = False) -> list[dict]:
    """Get the records for a given lithology name."""

    response = requests.get(f"{BASE_URL}/defs/lithologies?lith={lith_name}")
    response.raise_for_status()
    matches = response.json()["success"]["data"]
    if exact:
        matches = [match for match in matches if match["name"] == lith_name]
    if not matches:
        raise ValueError(f"No lithology found for '{lith_name}'")
    return matches


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
