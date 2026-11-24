"""Viewer disconnects and the profile's explicit-release lease policy."""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from hermes_cli.web_routers import display
from tools.bot_desktop import lease
from tests.bot_desktop_lease import reset_lease


class _Ws:
    """Just enough of a Starlette WebSocket: one disconnect message with the given close code."""

    def __init__(self, close_code: int):
        self._code = close_code
        self.closed = False

    async def accept(self):
        pass

    async def receive(self):
        await asyncio.sleep(0.05)
        return {"type": "websocket.disconnect", "code": self._code}

    async def send_bytes(self, data):
        pass

    async def close(self, code=1000, reason=""):
        self.closed = True


async def _bridge_once(close_code: int, home: str) -> lease.Lease:
    sock_dir = os.path.join(home, "bot-desktop")
    os.makedirs(sock_dir, exist_ok=True)
    sock = os.path.join(sock_dir, "rfb.sock")

    async def _xvnc(reader, writer):  # a silent framebuffer
        await asyncio.sleep(1)
        writer.close()

    server = await asyncio.start_unix_server(_xvnc, path=sock)
    try:
        info = {"hermes_home": home, "viewer_id": "desk-1"}
        await display._bridge(_Ws(close_code), info)
    finally:
        server.close()
    return lease.get(profile_key=home)


# 1005 (no status code) is what noVNC's code-less socket.close() AND some proxies produce on a drop, so
# the server keeps the lease; the Desktop sends an explicit 1000 when the pane is closed on purpose.
@pytest.mark.parametrize(("close_code", "human_keeps_control"), [
    (1006, True), (1005, True), (1000, False), (1001, False),
])
def test_only_a_clean_viewer_close_hands_the_screen_back(monkeypatch, close_code, human_keeps_control):
    reset_lease()
    with tempfile.TemporaryDirectory() as home:
        lease.acquire("desk-1", profile_key=home)
        after = asyncio.run(_bridge_once(close_code, home))
    reset_lease()
    assert (after.holder == lease.HUMAN) is human_keeps_control, after


@pytest.mark.parametrize("close_code", [1000, 1001, 1005, 1006])
def test_explicit_release_survives_disconnect_and_reconnect(tmp_path, monkeypatch, close_code):
    home = tmp_path / "worker"
    home.mkdir()
    (home / "config.yaml").write_text("bot_desktop:\n  explicit_release: true\n", encoding="utf-8")

    class Writer:
        def close(self):
            pass

        async def wait_closed(self):
            pass

    async def open_rfb(profile_home):
        assert profile_home == home
        return asyncio.StreamReader(), Writer(), None

    monkeypatch.setattr(display, "_open_rfb", open_rfb)
    acquired = lease.acquire("desk-1", profile_key=str(home))
    # A fresh configuration read after a gateway restart must not silently clear the saved hold.
    (home / "config.yaml").unlink()

    async def reconnect():
        for _ in range(2):
            await display._bridge(_Ws(close_code), {"hermes_home": str(home), "viewer_id": "desk-1"})
            assert lease.get(profile_key=str(home)) == acquired

    asyncio.run(reconnect())
    assert lease.get(profile_key=str(home)).holder == lease.HUMAN
    assert lease.release("desk-1", profile_key=str(home)).holder == lease.AGENT


def test_ex_holder_who_handed_back_is_not_evicted_by_a_later_takeover():
    """desk-1 holds, hands back to the agent, then desk-2 takes over: desk-1 is a plain watcher again
    and must stay connected; only a takeover WHILE desk-1 held (or believed it held) kicks it."""
    held = {"ever": False}
    assert display._should_evict(held, lease.Lease(holder=lease.HUMAN, viewer_id="desk-1"), "desk-1") is False
    assert display._should_evict(held, lease.Lease(holder=lease.AGENT), "desk-1") is False
    assert display._should_evict(held, lease.Lease(holder=lease.HUMAN, viewer_id="desk-2"), "desk-1") is False

    held = {"ever": False}
    display._should_evict(held, lease.Lease(holder=lease.HUMAN, viewer_id="desk-1"), "desk-1")
    assert display._should_evict(held, lease.Lease(holder=lease.HUMAN, viewer_id="desk-2"), "desk-1") is True


class _OpenWs(_Ws):
    """Stays open until ``finish`` is set, then reports a clean close."""

    def __init__(self):
        super().__init__(1000)
        self.finish = asyncio.Event()

    async def receive(self):
        await self.finish.wait()
        return {"type": "websocket.disconnect", "code": self._code}


def test_a_takeover_made_by_another_process_stops_input_within_the_refresh_interval(monkeypatch):
    """The bridge caches the input decision instead of reading lease.json per message; a takeover
    written by ANOTHER process (no in-process listener fires) must still be seen quickly."""
    from tools.bot_desktop import rfb_filter
    captured = {}

    class _Filter(rfb_filter.RfbClientFilter):
        def __init__(self, allow_input):
            super().__init__(allow_input)
            captured["allow"] = allow_input
    monkeypatch.setattr(rfb_filter, "RfbClientFilter", _Filter)

    async def _run(home: str) -> float:
        sock_dir = os.path.join(home, "bot-desktop")
        os.makedirs(sock_dir, exist_ok=True)
        server = await asyncio.start_unix_server(lambda r, w: None, path=os.path.join(sock_dir, "rfb.sock"))
        ws = _OpenWs()
        task = asyncio.create_task(display._bridge(ws, {"hermes_home": home, "viewer_id": "desk-1"}))
        try:
            while "allow" not in captured:
                await asyncio.sleep(0.01)
            assert captured["allow"]() is True
            # Another process takes over: the file changes, no listener in this process is told.
            lease._write(lease._path(home), lease.Lease(holder=lease.HUMAN, viewer_id="desk-2", epoch=2))
            t0 = asyncio.get_running_loop().time()
            while captured["allow"]() and asyncio.get_running_loop().time() - t0 < 2.0:
                await asyncio.sleep(0.02)
            return asyncio.get_running_loop().time() - t0
        finally:
            ws.finish.set()
            await task
            server.close()

    reset_lease()
    with tempfile.TemporaryDirectory() as home:
        lease.acquire("desk-1", profile_key=home)
        elapsed = asyncio.run(_run(home))
    reset_lease()
    assert elapsed < 0.5, elapsed


def test_no_bridge_task_or_socket_outlives_the_bridge():
    """The pumps were cancelled but never awaited and the Xvnc writer closed without wait_closed, so
    a returned _bridge still had tasks (and a half-closed socket) in flight on the loop; a viewer
    reconnecting in a tight loop accumulated them. Everything the bridge started is finished, and
    Xvnc has seen EOF, when it returns."""
    async def _run(home: str) -> set:
        sock_dir = os.path.join(home, "bot-desktop")
        os.makedirs(sock_dir, exist_ok=True)
        xvnc_saw_eof = asyncio.Event()

        async def _xvnc(reader, writer):
            await reader.read()  # until the bridge closes its end
            xvnc_saw_eof.set()

        server = await asyncio.start_unix_server(_xvnc, path=os.path.join(sock_dir, "rfb.sock"))
        try:
            await display._bridge(_Ws(1000), {"hermes_home": home, "viewer_id": "desk-1"})
            leftover = {t for t in asyncio.all_tasks()
                        if not t.done() and t.get_coro().__qualname__.startswith("_bridge.")}
            await asyncio.wait_for(xvnc_saw_eof.wait(), 2)
            return leftover
        finally:
            server.close()

    reset_lease()
    with tempfile.TemporaryDirectory() as home:
        leftover = asyncio.run(_run(home))
    assert leftover == set(), [t.get_coro() for t in leftover]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("explicit", [False, True])
def test_relay_queue_drain_does_not_acknowledge_remote_input(tmp_path, monkeypatch, explicit):
    from types import SimpleNamespace

    home = str(tmp_path)
    if explicit:
        (tmp_path / "config.yaml").write_text("bot_desktop:\n  explicit_release: true\n")
    keypress = b"\x04\x01\x00\x00\x00\x00\x00a"

    async def run():
        packets = asyncio.Queue()
        writes = asyncio.Queue()
        drained = asyncio.Event()
        applied = asyncio.Event()
        allow_remote = asyncio.Event()

        async def remote():
            await allow_remote.wait()
            assert (await writes.get()).endswith(keypress)
            applied.set()

        class Ws(_Ws):
            async def receive(self):
                return await packets.get()

        class RelayWriter:
            def write(self, data):
                writes.put_nowait(data)

            async def drain(self):
                drained.set()

            def close(self):
                pass

            async def wait_closed(self):
                pass

        async def open_rfb(_home):
            return asyncio.StreamReader(), RelayWriter(), SimpleNamespace(kill=lambda: None)

        monkeypatch.setattr(display, "_open_rfb", open_rfb)
        lease.acquire("desk-1", profile_key=home)
        task = asyncio.create_task(display._bridge(Ws(1006), {"hermes_home": home, "viewer_id": "desk-1"}))
        executor = asyncio.create_task(remote())
        try:
            await packets.put({"bytes": b"RFB 003.008\n\x01\x00" + keypress})
            await asyncio.wait_for(drained.wait(), 2)
            assert not applied.is_set()
            if explicit:
                assert lease.acquire("desk-2", profile_key=home).viewer_id == "desk-1"
                assert lease.viewer_may_send_input("desk-1", profile_key=home)
            released = lease.release("desk-1", profile_key=home)
            assert (released.holder == lease.HUMAN) is explicit
            if explicit:
                assert released.handback_pending
            assert not lease.viewer_may_send_input("desk-1", profile_key=home)
            assert not lease.viewer_may_send_input("desk-2", profile_key=home)
            allow_remote.set()  # the effect lands on the executor after local drain and hand-back
            await asyncio.wait_for(applied.wait(), 2)
            await packets.put({"type": "websocket.disconnect", "code": 1006})
            await asyncio.wait_for(task, 2)
            if explicit:
                assert lease.release("desk-1", profile_key=home).handback_pending
                assert lease.acquire("desk-2", profile_key=home).pending
                with pytest.raises(lease.HumanHasControl):
                    lease.assert_agent_may_act(home)
        finally:
            task.cancel()
            executor.cancel()
            await asyncio.gather(task, executor, return_exceptions=True)

    asyncio.run(run())
