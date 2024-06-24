import os
import httpx
import logging
from dotenv import load_dotenv
from json import JSONDecodeError

from text2graph.apiutils import sanitize_string


logger = logging.getLogger()
load_dotenv()
GEOCODE_API_BASE_URL = "https://geocode.maps.co/search?"


async def get_gps(
    query: str, client: httpx.AsyncClient, fail_attempts_max: int = 10
) -> tuple[float, float, str] | tuple[None, None, str]:
    """Get GPS coordinates from geocode api for a location query."""

    response, request_url_no_key = await make_geocode_query(query=query, client=client)
    fail_attempts_counter = 0
    while response.status_code in [
        "429",
        "503",
    ]:  # throttle warning or service unavailable
        logging.warning(
            f"Location hydrate geocode api request exceeded API limit or Service Unavailable for {query=}: {response.status_code=} {response.content=}"
        )
        if fail_attempts_counter > fail_attempts_max:
            logging.warning(
                f"Exceeded max retry attempts ({fail_attempts_max}) for {query=}: {response.status_code=} {response.content=}"
            )
            return None, None, request_url_no_key
        fail_attempts_counter += 1
        logger.warning(
            "f{client!r} recieved status {response.status_code}, backing off rate."
        )
        client.interval_back_off(multiplier=2)
        response, request_url_no_key = make_geocode_query(query=query, client=client)

    client.reduce_interval_if_last_ten_ok()

    try:
        lat = float(response.json()[0]["lat"])
        lon = float(response.json()[0]["lon"])
        return lat, lon, request_url_no_key
    except (KeyError, IndexError, JSONDecodeError):
        logging.warning(
            f"Location hydrate geocode api request failed for {query=}: {response.status_code=} {response.content=}"
        )
        return None, None, request_url_no_key


async def make_geocode_query(
    query: str, client: httpx.AsyncClient
) -> tuple[httpx.Response, str]:
    sanitized_query = sanitize_string(query)
    geocode_api_key = os.environ["GEOCODE_API_KEY"]
    request_url_no_key = f"{GEOCODE_API_BASE_URL}&q={sanitized_query}"
    request_url = request_url_no_key + f"&api_key={geocode_api_key}"
    response = await client.get(request_url)
    assert response.status_code != 403  # https://geocode.maps.co/
    client.track_response_codes(response.status_code)
    return response, request_url_no_key
