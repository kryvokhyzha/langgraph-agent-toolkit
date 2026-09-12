"""Check load-test scope guards and history verdicts without a server."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from db_pressure import (
    CancellableAsyncConnection,
    LoadDatabase,
    hold_checkpoint_table_lock,
    terminate_toolkit_sessions,
    verify_http_history,
)


def target(**changes):
    return LoadDatabase(
        **{
            "dsn": "postgresql://test@127.0.0.1:18516/lat_test",
            "schema": "lat_load_unit",
            "application_prefix": "lat-load-unit",
            **changes,
        }
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"dsn": "postgresql://test@192.0.2.1/lat_test"},
        {"dsn": "host=127.0.0.1 hostaddr=192.0.2.1 dbname=lat_test"},
        {"dsn": "host=/tmp dbname=lat_test"},
        {"dsn": "host=127.0.0.1 dbname=lat_test service=production"},
        {"schema": "public"},
        {"application_prefix": "production"},
    ],
)
def test_scope_guard_rejects_unsafe_targets(monkeypatch, changes):
    monkeypatch.setenv("LAT_LOAD_TEST_DATABASE", "yes")
    with pytest.raises(ValueError):
        target(**changes).validate()


@pytest.mark.parametrize("operation", ["lock", "terminate"])
async def test_mutation_requires_opt_in_before_connecting(monkeypatch, operation):
    monkeypatch.delenv("LAT_LOAD_TEST_DATABASE", raising=False)
    connect = AsyncMock(side_effect=AssertionError("The scope guard did not run."))
    monkeypatch.setattr(CancellableAsyncConnection, "connect", connect)
    with pytest.raises(ValueError, match="LAT_LOAD_TEST_DATABASE=yes"):
        if operation == "lock":
            await hold_checkpoint_table_lock(target(), asyncio.Event())
        else:
            await terminate_toolkit_sessions(target())
    connect.assert_not_awaited()


def test_scope_guard_accepts_loopback_and_excludes_observer_roles(monkeypatch):
    monkeypatch.setenv("LAT_LOAD_TEST_DATABASE", "yes")
    database = target(dsn="postgresql://test:private-test-value@[::1]:18516/lat_test")
    database.validate()
    assert "private-test-value" not in repr(database)
    assert database.application_names(("locks", "saver")) == ["lat-load-unit-locks", "lat-load-unit-saver"]
    with pytest.raises(ValueError, match="pool roles"):
        database.application_names(("observer",))


async def test_history_verdict_accepts_serialized_completion_order_across_pages():
    messages = [
        {"type": "human", "content": "second"},
        {"type": "ai", "content": "reply-second"},
        {"type": "human", "content": "first"},
        {"type": "ai", "content": "reply-first"},
    ]

    def response(request):
        assert request.url.path == "/load-agent/history"
        assert request.url.params["thread_id"] == "shared"
        offset = int(request.url.params["offset"])
        return httpx.Response(
            200, json={"messages": messages[offset : offset + 2], "total": 4, "next_offset": 2 if offset == 0 else None}
        )

    async with httpx.AsyncClient(base_url="http://127.0.0.1", transport=httpx.MockTransport(response)) as client:
        report = await verify_http_history(
            client,
            "load-agent",
            "shared",
            ["first", "second"],
            expected_ai_messages=["reply-first", "reply-second"],
            expected_pairs={"first": "reply-first", "second": "reply-second"},
        )
    assert report["ok"]
    assert report["human_order"] == ["second", "first"]
    assert report["total"] == 4


async def test_history_verdict_reports_missing_duplicate_and_mispaired_turns():
    messages = [
        {"type": "human", "content": "first"},
        {"type": "ai", "content": "reply-second"},
        {"type": "human", "content": "first"},
        {"type": "ai", "content": "reply-first"},
    ]
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"messages": messages, "total": 4})),
    ) as client:
        report = await verify_http_history(
            client,
            "load-agent",
            "shared",
            ["first", "second"],
            expected_pairs={"first": "reply-first", "second": "reply-second"},
        )
    assert not report["ok"]
    assert report["human"]["missing"] == ["second"]
    assert report["human"]["duplicates"] == {"first": 2}
    assert report["reply_order_errors"] == ["first"]


async def test_history_rejects_pagination_that_does_not_advance():
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"messages": [], "total": 2, "next_offset": 0})
        ),
    ) as client:
        with pytest.raises(ValueError, match="did not advance"):
            await verify_http_history(client, "load-agent", "shared", [])


async def test_history_accepts_expected_repeated_reply_content():
    messages = [
        {"type": "human", "content": "first"},
        {"type": "ai", "content": "OK"},
        {"type": "human", "content": "second"},
        {"type": "ai", "content": "OK"},
    ]
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"messages": messages, "total": 4})),
    ) as client:
        report = await verify_http_history(
            client,
            "load-agent",
            "shared",
            ["first", "second"],
            expected_ai_messages=["OK", "OK"],
            expected_pairs={"first": "OK", "second": "OK"},
        )
    assert report["ok"]
    assert report["ai"]["duplicates"] == {}
