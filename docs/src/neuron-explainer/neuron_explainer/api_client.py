import asyncio
import contextlib
import os
import random
import traceback
from asyncio import Semaphore
from functools import wraps
from typing import Any, Callable, Optional

import httpx
import orjson


def is_api_error(err: Exception) -> bool:
    if isinstance(err, httpx.HTTPStatusError):
        response = err.response
        error_data = response.json().get("error", {})
        error_message = error_data.get("message")
        if response.status_code in [400, 404, 415]:
            if error_data.get("type") == "idempotency_error":
                print(f"Retrying after idempotency error: {error_message} ({response.url})")
                return True
            else:
                # Invalid request
                return False
        else:
            print(f"Retrying after API error: {error_message} ({response.url})")
            return True

    elif isinstance(err, httpx.ConnectError):
        print(f"Retrying after connection error... ({err.request.url})")
        return True

    elif isinstance(err, httpx.TimeoutException):
        print(f"Retrying after a timeout error... ({err.request.url})")
        return True

    elif isinstance(err, httpx.ReadError):
        print(f"Retrying after a read error... ({err.request.url})")
        return True

    print(f"Retrying after an unexpected error: {repr(err)}")
    traceback.print_tb(err.__traceback__)
    return True


def exponential_backoff(
    retry_on: Callable[[Exception], bool] = lambda err: True
) -> Callable[[Callable], Callable]:
    """
    Returns a decorator which retries the wrapped function as long as the specified retry_on
    function returns True for the exception, applying exponential backoff with jitter after
    failures, up to a retry limit.
    """
    init_delay_s = 1.0
    max_delay_s = 10.0
    # Roughly 30 minutes before we give up.
    max_tries = 200
    backoff_multiplier = 2.0
    jitter = 0.2

    def decorate(f: Callable) -> Callable:
        assert asyncio.iscoroutinefunction(f)

        @wraps(f)
        async def f_retry(*args: Any, **kwargs: Any) -> None:
            delay_s = init_delay_s
            for i in range(max_tries):
                try:
                    return await f(*args, **kwargs)
                except Exception as err:
                    if not retry_on(err) or i == max_tries - 1:
                        raise
                    jittered_delay = random.uniform(delay_s * (1 - jitter), delay_s * (1 + jitter))
                    await asyncio.sleep(jittered_delay)
                    delay_s = min(delay_s * backoff_multiplier, max_delay_s)

        return f_retry

    return decorate


API_KEY = os.getenv("OPENAI_API_KEY")
assert API_KEY, "Please set the OPENAI_API_KEY environment variable"
API_HTTP_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + API_KEY,
}
BASE_API_URL = "https://api.openai.com/v1"


class ApiClient:
    """Performs inference using the OpenAI API. Supports response caching and concurrency limits."""

    def __init__(
        self,
        model_name: str,
        # If set, no more than this number of HTTP requests will be made concurrently.
        max_concurrent: Optional[int] = None,
        # Whether to cache request/response pairs in memory to avoid duplicating requests.
        cache: bool = False,
    ):
        self.model_name = model_name

        if max_concurrent is not None:
            self._concurrency_check: Optional[Semaphore] = Semaphore(max_concurrent)
        else:
            self._concurrency_check = None

        if cache:
            self._cache: Optional[dict[str, Any]] = {}
        else:
            self._cache = None

    @exponential_backoff(retry_on=is_api_error)
    async def make_request(
        self, timeout_seconds: Optional[int] = None, **kwargs: Any
    ) -> dict[str, Any]:
        if self._cache is not None:
            key = orjson.dumps(kwargs)
            if key in self._cache:
                return self._cache[key]
        async with contextlib.AsyncExitStack() as stack:
            if self._concurrency_check is not None:
                await stack.enter_async_context(self._concurrency_check)
            http_client = await stack.enter_async_context(
                httpx.AsyncClient(timeout=timeout_seconds)
            )
            # If the request has a "messages" key, it should be sent to the /chat/completions
            # endpoint. Otherwise, it should be sent to the /completions endpoint.
            url = BASE_API_URL + ("/chat/completions" if "messages" in kwargs else "/completions")
            kwargs["model"] = self.model_name
            response = await http_client.post(url, headers=API_HTTP_HEADERS, json=kwargs)
        # The response json has useful information but the exception doesn't include it, so print it
        # out then reraise.
        try:
            response.raise_for_status()
        except Exception as e:
            print(response.json())
            raise e
        if self._cache is not None:
            self._cache[key] = response.json()
        return response.json()


if __name__ == "__main__":

    async def main() -> None:
        client = ApiClient(model_name="gpt-3.5-turbo", max_concurrent=1)
        print(await client.make_request(prompt="Why did the chicken cross the road?", max_tokens=9))

    asyncio.run(main())
