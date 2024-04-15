import logging
import os

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

load_dotenv()

SERP_API_BASE_URL = "https://serpapi.com/search.json?engine=google_maps"


async def get_gps(query: str) -> tuple[float, float, str] | tuple[None, None, str]:
    """Get GPS coordinates from serpapi for a location query."""

    serpapi_key = os.environ["SERPAPI_KEY"]
    google_maps_api_base_url = SERP_API_BASE_URL
    request_url_no_key = f"{google_maps_api_base_url}&q={query}"
    request_url = f"{google_maps_api_base_url}&q={query}&api_key={serpapi_key}"

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(request_url)

    response.raise_for_status()

    try:
        gps = response.json()["place_results"]["gps_coordinates"]
        return gps["latitude"], gps["longitude"], request_url_no_key
    except (KeyError, ValidationError):
        logging.warning(
            f"Location hydrate serpapi request failed for {query}: {response.status_code=} {response.content=}"
        )
        return None, None, request_url_no_key
