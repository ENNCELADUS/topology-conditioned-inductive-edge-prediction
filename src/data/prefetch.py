"""Bounded deterministic CPU prefetch for training and scoring."""

from __future__ import annotations

from collections import deque
from collections.abc import Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

_BatchT = TypeVar("_BatchT")


def _prefetch_batches(batches: Iterator[_BatchT], *, depth: int) -> Generator[_BatchT, None, None]:
    """Build bounded deterministic CPU batches ahead of GPU consumption."""
    if depth <= 0:
        yield from batches
        return

    iterator = iter(batches)

    def read_next() -> _BatchT:
        return next(iterator)

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="egostitch-batch")
    futures: deque[Future[_BatchT]] = deque(executor.submit(read_next) for _ in range(depth))
    try:
        while futures:
            future = futures.popleft()
            try:
                batch = future.result()
            except StopIteration:
                return
            futures.append(executor.submit(read_next))
            yield batch
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
