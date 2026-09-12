"""Read resource use for process IDs that the load harness owns."""

import asyncio
import math
import os
from contextlib import suppress


COMMAND_TIMEOUT = 2.0


def _process_ids(pids: list[int]) -> list[int]:
    if any(type(pid) is not int or not 0 < pid < 2**31 for pid in pids):
        raise ValueError("Process IDs must be positive integers below 2**31")
    return list(dict.fromkeys(pids))


async def _kill_and_reap(process: asyncio.subprocess.Process) -> None:
    with suppress(ProcessLookupError):
        process.kill()
    async with asyncio.timeout(COMMAND_TIMEOUT):
        await process.communicate()


async def _stop(process: asyncio.subprocess.Process) -> None:
    """Keep child cleanup active if the caller cancels again."""
    cleanup = asyncio.create_task(_kill_and_reap(process))
    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
    cleanup.result()
    if cancelled:
        raise asyncio.CancelledError


async def _output(*args: str) -> str | None:
    """Allow two seconds for a command and two seconds for forced cleanup."""
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env={**os.environ, "LC_ALL": "C"},
        )
    except OSError:
        return None
    try:
        async with asyncio.timeout(COMMAND_TIMEOUT):
            stdout, _ = await process.communicate()
    except (TimeoutError, asyncio.CancelledError) as error:
        await _stop(process)
        if isinstance(error, asyncio.CancelledError):
            raise
        return None
    # ps and lsof can return 1 when a requested process has exited.
    if process.returncode not in (0, 1):
        return None
    return stdout.decode("utf-8", errors="replace")


async def process_metrics(pids: list[int]) -> list[dict[str, int | float]]:
    """Return current RSS bytes and the CPU average reported by ps.

    The caller must supply process IDs that the load harness owns.
    Missing processes and failed samples have no result row.
    The CPU value is not CPU use during the last sample interval.
    """
    selected = _process_ids(pids)
    if not selected:
        return []
    output = await _output("ps", "-p", ",".join(map(str, selected)), "-o", "pid=,rss=,pcpu=")
    rows = {}
    for line in (output or "").splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            pid, rss, cpu = int(fields[0]), int(fields[1]), float(fields[2])
        except ValueError:
            continue
        if pid in selected and rss >= 0 and math.isfinite(cpu) and cpu >= 0:
            rows[pid] = {"pid": pid, "rss_bytes": rss * 1024, "cpu_percent_ps_average": cpu}
    return [rows[pid] for pid in selected if pid in rows]


async def descriptor_counts(pids: list[int]) -> dict[int, int]:
    """Count numeric file descriptors from lsof for each observed process.

    The caller must supply process IDs that the load harness owns.
    Missing processes and unavailable lsof output have no result entry.
    Working directories, executable files, and memory maps are not descriptors.
    """
    selected = _process_ids(pids)
    if not selected:
        return {}
    output = await _output("lsof", "-n", "-P", "-p", ",".join(map(str, selected)), "-Fpf")
    descriptors: dict[int, set[int]] = {}
    current = None
    for line in (output or "").splitlines():
        if line.startswith("p"):
            current = int(line[1:]) if line[1:].isascii() and line[1:].isdecimal() else None
            if current in selected:
                descriptors.setdefault(current, set())
        elif current in descriptors and line.startswith("f") and line[1:].isascii() and line[1:].isdecimal():
            descriptors[current].add(int(line[1:]))
    return {pid: len(descriptors[pid]) for pid in selected if pid in descriptors}
