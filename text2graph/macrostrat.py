from functools import cache
import requests

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
