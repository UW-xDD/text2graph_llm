import re
import asyncio
import logging
import datetime as dt
from typing import Union
from functools import wraps
from httpx import AsyncClient


logger = logging.getLogger(__name__)

# unless you keep a strong reference to a running task, it can be dropped during execution
# https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
_background_tasks = set()


class RateLimitedClient(AsyncClient):
    """httpx.AsyncClient with a rate limit."""

    def __init__(
        self,
        interval: Union[dt.timedelta, float],
        count=1,
        label: str | None = None,
        **kwargs,
    ):
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

        self.count = count
        self.semaphore = asyncio.Semaphore(self.count)
        self.interval = interval
        self.min_interval = interval
        self.last_ten_send_times: list[dt.datetime] = []
        self.last_ten_response_codes: list[int] = []
        if not label:
            label = "RateLimitedClient"
        self.label = label
        super().__init__(**kwargs)

    def __repr__(self):
        cls = self.__class__.__name__
        ks = ["interval", "count", "label", "timeout"]
        self_dict_strs = [
            f"{k!r}={v!r}".replace("'", "") for k, v in self.__dict__.items() if k in ks
        ]
        return f"{cls}({', '.join(self_dict_strs)})"
        # return f"{cls}(interval={self.interval!r}, count={self.count!r})"

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
        self.track_send_times()
        send = asyncio.create_task(super().send(*args, **kwargs))
        self._schedule_semaphore_release()
        return await send

    def track_send_times(self):
        self.last_ten_send_times.append(dt.datetime.now())
        self.last_ten_send_times = self.last_ten_send_times[-10:]

    def track_response_codes(self, code: int):
        self.last_ten_response_codes.append(code)
        self.last_ten_response_codes = self.last_ten_response_codes[-10:]

    def last_ten_status_codes_ok(self) -> bool:
        if all([x == 200 for x in self.last_ten_response_codes]):
            return True
        else:
            return False

    def reduce_interval_if_last_ten_ok(self) -> None:
        if self.last_ten_status_codes_ok() and self.interval >= self.min_interval:
            self.interval_back_off(multiplier=0.5)

    def interval_back_off(self, multiplier):
        updated_interval = self.interval * multiplier
        if updated_interval >= self.min_interval:
            self.interval = updated_interval
        else:
            self.interval = self.min_interval
        logger.info(f"{self!r} interval now set to: {self.interval}s")


def sanitize_string(query: str) -> str:
    """convert non URL-able string chars to '-'"""
    return re.sub(r"[^0-9A-Za-z\.,\-\_ ]", "-", query)
