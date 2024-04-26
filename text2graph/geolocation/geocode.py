import os
import httpx
import asyncio
import logging
import datetime as dt
from typing import Union
from functools import wraps
from httpx import AsyncClient
from dotenv import load_dotenv
from json import JSONDecodeError

load_dotenv()
GEOCODE_API_BASE_URL = "https://geocode.maps.co/search?"
# unless you keep a strong reference to a running task, it can be dropped during execution
# https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
_background_tasks = set()


class RateLimitedClient(AsyncClient):
    """httpx.AsyncClient with a rate limit."""

    def __init__(self, interval: Union[dt.timedelta, float], count=1, **kwargs):
        """
        Parameters
        ----------
        interval : Union[dt.timedelta, float]
            Length of interval.
            If a float is given, seconds are assumed.
        numerator : int, optional
            Number of requests which can be sent in any given interval (default 1).
        """
        if isinstance(interval, dt.timedelta):
            interval = interval.total_seconds()

        self.interval = interval
        self.semaphore = asyncio.Semaphore(count)
        super().__init__(**kwargs)

    def _schedule_semaphore_release(self):
        wait = asyncio.create_task(asyncio.sleep(self.interval))
        _background_tasks.add(wait)

        def wait_cb(task):
            self.semaphore.release()
            _background_tasks.discard(task)

        wait.add_done_callback(wait_cb)

    @wraps(AsyncClient.send)
    async def send(self, *args, **kwargs):
        await self.semaphore.acquire()
        send = asyncio.create_task(super().send(*args, **kwargs))
        self._schedule_semaphore_release()
        return await send


async def get_gps(
    query: str, client: httpx.AsyncClient
) -> tuple[float, float, str] | tuple[None, None, str]:
    """Get GPS coordinates from geocode api for a location query."""

    geocode_api_key = os.environ["GEOCODE_API_KEY"]
    request_url_no_key = f"{GEOCODE_API_BASE_URL}&q={query}"
    request_url = request_url_no_key + f"&api_key={geocode_api_key}"
    response = await client.get(request_url)
    try:
        lat = response.json()[0]["lat"]
        lon = response.json()[0]["lon"]
        return lat, lon, request_url_no_key
    except (KeyError, IndexError, JSONDecodeError):
        logging.warning(
            f"Location hydrate geocode api request failed for {query}: {response.status_code=} {response.content=}"
        )
        return None, None, request_url_no_key
