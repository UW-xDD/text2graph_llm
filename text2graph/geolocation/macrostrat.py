import json
import logging

import requests
from pathlib import Path
from tqdm.auto import tqdm

import text2graph


def all_strat_names_long() -> list[dict[str, str | int | float]] | None:
    """
    fetch all strat name from macrostrat
    :return: list of strat_name dicts
    """
    url = "https://macrostrat.org/api/defs/strat_names?all&response=long"
    r = requests.get(url)
    result = None
    if r.status_code == 200:
        result = r.json()["success"]["data"]
    return result


def macrostrat_units(strat_name_ids: list[int]) -> list[dict[str, str | int | float]]:
    """
    fetch all units associatd with strat_name_id
    :param strat_name_ids: list of strat_name_ids to return unit records for
    :return: strat_name_dict with coords or empty coords key
    """
    strat_name_ids = [str(x) for x in strat_name_ids]
    url = f"https://macrostrat.org/api/units?strat_name_id={','.join(strat_name_ids)}&response=long"
    r = requests.get(url)
    strat_name_units = []
    try:
        strat_name_units = r.json()["success"]["data"]
    except KeyError:
        pass
    return strat_name_units


def local_stratname_records() -> list[dict[str : str | int | float]]:
    local_strat_name_data = (
        Path(text2graph.__file__).parent.parent
        / "data"
        / "macrostrat_stratname_data.json"
    )
    if not local_strat_name_data.exists():
        local_strat_name_data.parent.parent.mkdir(exist_ok=True)
        api_json_response = all_strat_names_long()
        strat_name_ids_with_unit_records = [
            x["strat_name_id"] for x in api_json_response if x["t_units"]
        ]
        pbar = tqdm(total=len(strat_name_ids_with_unit_records))
        pbar.set_description("fetching macrostrat stratname records")
        records = []
        for ten_strat_ids in zip(*[iter(strat_name_ids_with_unit_records)] * 10):
            records.append(macrostrat_units(ten_strat_ids))
            pbar.update(10)

        flat_records = [y for x in records for y in x]
        with open(local_strat_name_data, "w") as f:
            json.dump(flat_records, f)

    with open(local_strat_name_data, "r") as f:
        strat_name_records = json.load(f)

    return strat_name_records


class StratNameGPSLookup:
    def __init__(self):
        """
        provide strat_name_long and strat_name_id : lat, lon lookup
        """
        self.api_json_response = local_stratname_records()
        self.lookup = {}
        for x in self.api_json_response:
            try:
                entry = dict(
                    name=x["strat_name_long"],
                    lat=float(x["clat"]),
                    lon=float(x["clng"]),
                )
                self.lookup[x["strat_name_long"]] = entry
                self.lookup[x["strat_name_id"]] = entry
            except (KeyError, ValueError):
                logging.warning(
                    f"invalid location: {x['strat_name_long']}, lat={x['clat']}, lon={x['clng']}"
                )
                pass

    def __call__(self, strat_name_or_id: str | int) -> dict[str, str | float] | None:
        try:
            d = self.lookup[strat_name_or_id]
        except KeyError:
            d = None
        return d
