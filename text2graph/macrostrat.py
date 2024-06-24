from functools import cache
import logging
import re

import httpx
import requests

from text2graph.apiutils import sanitize_string

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


@cache
def get_all_strat_names(long: bool = False) -> list[str]:
    """Get all stratigraphic names from macrostrat API."""

    url = f"{BASE_URL}/defs/strat_names?all"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()["success"]["data"]

    key = "strat_name_long" if long else "strat_name"
    return sorted(list(set([x[key] for x in data])))


def get_all_intervals() -> list[dict]:
    """Get all stratigraphic intervals from macrostrat API."""

    url = f"{BASE_URL}/defs/intervals?all"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()["success"]["data"]
    return data


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


async def get_strat_records(
    strat_name: str, client: httpx.AsyncClient, exact: bool = False
) -> list[dict]:
    """Get the records for a given stratigraphic name."""

    sanitized_stratname = sanitize_string(strat_name)
    response = await client.get(
        f"{BASE_URL}/defs/strat_names?strat_name={sanitized_stratname}"
    )
    try:
        response.raise_for_status()
        macrostrat_version = response.json()["success"]["v"]
        matches = response.json()["success"]["data"]
        for match in matches:
            match["macrostrat_version"] = macrostrat_version
        if exact:
            matches = [
                match for match in matches if match["strat_name"] == sanitized_stratname
            ]
        if not matches:
            logging.warning(
                f"No stratigraphic name found for {strat_name=}, {sanitized_stratname=}"
            )
    except Exception as e:
        logging.warning(
            f"Strat Record collection failure for {strat_name=}, {sanitized_stratname=} with: {e}"
        )
        matches = [{}]
    return matches


async def get_lith_records(lith_name: str, exact: bool = False) -> list[dict]:
    """Get the records for a given lithology name."""

    santitized_lith_name = sanitize_string(lith_name)

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/defs/lithologies?lith={santitized_lith_name}"
        )
        response.raise_for_status()
        matches = response.json()["success"]["data"]
        if exact:
            matches = [
                match for match in matches if match["name"] == santitized_lith_name
            ]
        if not matches:
            logging.warning(
                f"No lithology found for {lith_name=}, {santitized_lith_name=}"
            )
    return matches


def _find_word_occurrences(text: str, search_word: str) -> list[dict]:
    """Find all occurrences of a word in a given text and get its position of occurrence."""

    matches = [match for match in re.finditer(rf"\b{re.escape(search_word)}\b", text)]
    results = [
        {
            "word": match.group(),
            "start": match.start(),
            "end": match.end(),
            "link": f"{BASE_URL}/defs/autocomplete?query={match.group()}",
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
