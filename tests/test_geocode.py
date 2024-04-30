import asyncio
from datetime import timedelta
import pytest


from text2graph.geolocation.geocode import RateLimitedClient, get_gps


GEOCODE_RESPONSE = [{"lat": "43.074761", "lon": "-89.3837613"}]


@pytest.mark.asyncio
async def test_rate_limited_client_rate_respected(httpx_mock) -> None:
    httpx_mock.add_response(json=GEOCODE_RESPONSE)
    query = "Madison, WI"
    interval = timedelta(seconds=1.5)
    client = RateLimitedClient(interval=interval, count=1, timeout=30)
    _ = await asyncio.gather(*[get_gps(query=query, client=client) for _ in range(10)])
    deltas = [
        client.last_ten_send_times[i + 1] - x
        for i, x in enumerate(client.last_ten_send_times[:-1])
    ]
    assert all([delta >= interval for delta in deltas])


def test_get_gps(httpx_mock) -> None:
    httpx_mock.add_response(json=GEOCODE_RESPONSE)
    lat, lon, url = asyncio.run(
        get_gps(
            query="Madison, WI",
            client=RateLimitedClient(interval=1.5, count=1, timeout=30),
        )
    )
    assert lat == 43.074761
    assert lon == -89.3837613
    assert url == "https://geocode.maps.co/search?&q=Madison, WI"
