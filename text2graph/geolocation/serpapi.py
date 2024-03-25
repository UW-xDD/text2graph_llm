import os
import logging
import requests
from dotenv import load_dotenv
from pydantic import ValidationError


from text2graph.schema import Location


load_dotenv()


def serpapi_location_result(q: str) -> Location | None:
    serpapi_key = os.environ["SERPAPI_KEY"]
    google_maps_api_base_url = "https://serpapi.com/search.json?engine=google_maps"
    request_url = f"{google_maps_api_base_url}&q={q}&api_key={serpapi_key}"
    r = requests.get(request_url)
    p = None
    if r.ok:
        search_result = r.json()
        try:
            p = Location(
                name="",
                lat=search_result["place_results"]["gps_coordinates"]["latitude"],
                lon=search_result["place_results"]["gps_coordinates"]["longitude"],
            )
        except (KeyError, ValidationError):
            pass
    else:
        logging.warning(f"serpapi request failed: {r.status_code=} {r.content=}")
    return p
