"""Check process samples and child cleanup without a load run."""

import asyncio
import sys
from unittest.mock import AsyncMock

import pytest
import resources


async def test_process_metrics_convert_rss_and_omit_exited_processes(monkeypatch):
    command = AsyncMock(return_value="  812 2048 10.5\n 999 4096 0.0\n 813 512 nan\n invalid\n")
    monkeypatch.setattr(resources, "_output", command)
    assert await resources.process_metrics([812, 813, 814, 812]) == [
        {"pid": 812, "rss_bytes": 2097152, "cpu_percent_ps_average": 10.5}
    ]
    command.assert_awaited_once_with("ps", "-p", "812,813,814", "-o", "pid=,rss=,pcpu=")


async def test_descriptor_counts_exclude_non_descriptors_and_other_processes(monkeypatch):
    command = AsyncMock(return_value="p812\nfcwd\nftxt\nfmem\nf0\nf1\nf2\nf2\nf10\np999\nf3\np813\nf0\n")
    monkeypatch.setattr(resources, "_output", command)
    assert await resources.descriptor_counts([812, 813, 814]) == {812: 4, 813: 1}
    command.assert_awaited_once_with("lsof", "-n", "-P", "-p", "812,813,814", "-Fpf")


@pytest.mark.parametrize("pids", [[0], [-1], [True], ["812"], [2**31]])
async def test_reject_invalid_process_selection_before_command(monkeypatch, pids):
    command = AsyncMock()
    monkeypatch.setattr(resources, "_output", command)
    for sample in (resources.process_metrics, resources.descriptor_counts):
        with pytest.raises(ValueError, match="Process IDs"):
            await sample(pids)
    command.assert_not_awaited()


async def test_empty_selection_does_not_start_process(monkeypatch):
    spawn = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    assert await resources.process_metrics([]) == []
    assert await resources.descriptor_counts([]) == {}
    spawn.assert_not_awaited()


async def test_unavailable_commands_do_not_report_zero_usage(monkeypatch):
    spawn = AsyncMock(side_effect=FileNotFoundError)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    assert await resources.process_metrics([812]) == []
    assert await resources.descriptor_counts([812]) == {}


class StalledProcess:
    def __init__(self):
        self.returncode = None
        self.started = asyncio.Event()
        self.killed = asyncio.Event()
        self.reaped = asyncio.Event()
        self.allow_reap = asyncio.Event()
        self.allow_reap.set()

    async def communicate(self):
        self.started.set()
        await self.killed.wait()
        await self.allow_reap.wait()
        self.returncode = -9
        self.reaped.set()
        return b"", None

    def kill(self):
        self.killed.set()


async def test_timeout_kills_and_reaps_command(monkeypatch):
    process = StalledProcess()
    monkeypatch.setattr(resources, "COMMAND_TIMEOUT", 0.01)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    async with asyncio.timeout(1):
        assert await resources.process_metrics([812]) == []
    assert process.killed.is_set()
    assert process.reaped.is_set()


async def test_timeout_reaps_a_real_child_process(monkeypatch):
    spawn = asyncio.create_subprocess_exec
    children = []

    async def record_child(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(resources, "COMMAND_TIMEOUT", 0.1)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", record_child)
    async with asyncio.timeout(2):
        assert await resources._output(sys.executable, "-c", "import time; time.sleep(30)") is None
    assert len(children) == 1
    assert children[0].returncode is not None
    assert children[0].stdout.at_eof()


async def test_repeated_cancellation_still_reaps_command(monkeypatch):
    process = StalledProcess()
    process.allow_reap.clear()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    task = asyncio.create_task(resources.descriptor_counts([812]))
    async with asyncio.timeout(1):
        await process.started.wait()
        task.cancel()
        await process.killed.wait()
        task.cancel()
        await asyncio.sleep(0)
        process.allow_reap.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert process.reaped.is_set()


async def test_partial_process_list_survives_exit_status_one(monkeypatch):
    process = AsyncMock(returncode=1)
    process.communicate.return_value = (b"812 1024 0.0\n", None)
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    assert await resources.process_metrics([812, 813]) == [
        {"pid": 812, "rss_bytes": 1048576, "cpu_percent_ps_average": 0.0}
    ]
    assert spawn.call_args.kwargs["env"]["LC_ALL"] == "C"
    assert spawn.call_args.kwargs["stderr"] == asyncio.subprocess.DEVNULL
