import logging
import re
from enum import Enum
from functools import cache
from urllib.parse import quote

import httpx
import requests

from text2graph.utils import log_time

BASE_URL = "https://macrostrat.org/api"

ROUTES_DOCS = {
    "/defs/autocomplete": "Quickly retrieve all definitions matching a query. Limited to 100 results.",
    "/defs/define": "Define multiple terms simultaneously",
    "/defs/languages": "Returns ISO 639-3 and ISO 639-1 codes for all languages",
    "/defs/lithologies": "Returns all lithology definitions",
    "/defs/lithology_attributes": "Returns lithology attribute definitions",
    "/defs/structures": "Returns all structure definitions",
    "/defs/columns": "Returns column definitions",
    "/defs/econs": "Returns econ definitions",
    "/defs/environments": "Returns environment definitions",
    "/defs/intervals": "Returns all time interval definitions",
    "/defs/sources": "Returns sources associated with geologic units. If a geographic format is requested, the bounding box of the source is returned as the geometry.",
    "/defs/strat_names": "Returns strat names",
    "/defs/strat_name_concepts": "Returns strat name concepts",
    "/defs/timescales": "Returns timescales used by Macrostrat",
    "/defs/minerals": "Returns mineral names and formulas",
    "/defs/projects": "Returns available Macrostrat projects",
    "/defs/plates": "Returns definitions of plates from /paleogeography",
    "/defs/measurements": "Returns all measurements definitions",
    "/defs/groups": "Returns all column groups",
    "/defs/grainsizes": "Returns grain size definitions",
    "/defs/refs": "Returns references",
    "/defs/drilling_sites": "Returns metadata for offshore drilling sites from ODP, DSDP and IODP",
}


class EntityType(Enum):
    """Entity types that can be used in LLM."""

    STRAT_NAME = "strat_name"
    MINERAL = "mineral"
    LITHOLOGY = "lithology"


@cache
def get_all_strat_names(long: bool = False) -> list[str]:
    """Get all stratigraphic names from macrostrat API."""

    url = f"{BASE_URL}/defs/strat_names?all"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()["success"]["data"]

    key = "strat_name_long" if long else "strat_name"
    return sorted(list(set([x[key] for x in data])))


@cache
def get_all_mineral_names(lower: bool = True) -> list[str]:
    """Get all mineral names from macrostrat API."""

    url = f"{BASE_URL}/defs/minerals?all"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()["success"]["data"]
    names = sorted(list(set([x["mineral"] for x in data])))

    if not lower:
        return names
    return [name.lower() for name in names]


@cache
def get_all_intervals() -> list[dict]:
    """Get all stratigraphic intervals from macrostrat API."""

    url = f"{BASE_URL}/defs/intervals?all"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()["success"]["data"]
    return data


async def get_records(
    entity_type: EntityType, name: str, exact: bool = False
) -> list[dict]:
    """Get the records for a given stratigraphic name."""

    routes = {
        "strat_name": f"{BASE_URL}/defs/strat_names?strat_name={name}",
        "mineral": f"{BASE_URL}/defs/minerals?mineral={name}",
    }

    match_keys = {"strat_name": "strat_name", "mineral": "mineral"}

    async with httpx.AsyncClient() as client:
        response = await client.get(routes[entity_type.value])
        response.raise_for_status()
        macrostrat_version = response.json()["success"]["v"]
        matches = response.json()["success"]["data"]
        for match in matches:
            match["macrostrat_version"] = macrostrat_version

        if exact:
            matches = [
                match
                for match in matches
                if match[match_keys[entity_type.value]] == name
            ]
        if not matches:
            logging.warning(f"No stratigraphic name found for '{name}'")
    return matches


def _find_word_occurrences(text: str, word: str) -> list[dict]:
    """Find all occurrences of a word in a given text and get its position of occurrence."""

    matches = [match for match in re.finditer(rf"\b{re.escape(word)}\b", text)]
    results = [
        {
            "word": match.group(),
            "start": match.start(),
            "end": match.end(),
            "link": quote(
                f"{BASE_URL}/defs/autocomplete?query={match.group()}", safe=":/?="
            ),
        }
        for match in matches
    ]
    return results


@log_time
def find_all_occurrences(
    text: str, words: list[str], ignore_case: bool = False
) -> list[dict[str, str | int]]:
    """Find all occurrences of a list of terms in a given text and get its position of occurrence."""

    if ignore_case:
        words = [word.lower() for word in words]
        text = text.lower()

    occurrences = []
    for word in words:
        if word not in text:
            continue
        this_occ = _find_word_occurrences(text=text, word=word)
        if this_occ:
            occurrences.extend(this_occ)

    return sorted(occurrences, key=lambda x: x["start"])
