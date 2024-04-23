import os
import httpx
import logging
from dotenv import load_dotenv


load_dotenv()
GEOCODE_API_BASE_URL = "https://geocode.maps.co/search?"


async def get_gps(query: str) -> tuple[float, float, str] | tuple[None, None, str]:
    """Get GPS coordinates from geocode api for a location query."""

    geocode_api_key = os.environ["GEOCODE_API_KEY"]
    request_url_no_key = f"{GEOCODE_API_BASE_URL}&q={query}"
    request_url = request_url_no_key + f"&api_key={geocode_api_key}"

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(request_url)

    response.raise_for_status()

    try:
        lat = response.json()[0]["lat"]
        lon = response.json()[0]["lon"]
        return lat, lon, request_url_no_key
    except KeyError:
        logging.warning(
            f"Location hydrate geocode api request failed for {query}: {response.status_code=} {response.content=}"
        )
        return None, None, request_url_no_key
