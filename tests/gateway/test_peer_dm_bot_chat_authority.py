"""A peer turn into the canonical Bot Chat under session authority is that chat's own admission.

``hermes peer dm`` / ``peer run`` reach the API edge; under the unified runtime the Bot Chat is
a LOCAL session whose only writer is the authority FIFO, so the turn goes through the
``bot_relay.deliver`` door (``gateway/session_bot.py``) and the reply is its admission's
receipt. Binding the chat as an API conversation would be a second writer (``permission_denied``).
"""
import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.session_contract import SessionActor
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session
from tests.waiting import async_wait_until

AUTHOR = {"id": "bot:cto", "name": "cto", "is_bot": True}


@pytest_asyncio.fixture
async def peer_runtime(tmp_path, monkeypatch, api_store, request):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_local import create_local_session
    from gateway.session_local_title import title_new_session

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    monkeypatch.setattr('tools.bot_mode_dm._LIVE_WAIT_SECONDS', 2.0)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store.persistence, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, persistence=api_store(store.persistence))
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='fixture')
    owner = SessionActor(getattr(request, 'param', 'uid:1000'), authority.profile_id,
                         frozenset({'session:create', 'session:read', 'session:submit'}), 'native')
    chat = create_local_session(authority, owner, {'request_id': 'bot', 'source': 'gui', 'cwd': str(tmp_path),
                                                    'model': 'fixture', 'toolsets': []})
    title_new_session(authority, chat, 'Bot Chat')
    seed_session(authority.persistence, 'scratch', 'api_server')
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    adapter._home_stores[str(tmp_path)] = authority.persistence
    from gateway.config import Platform
    runner.adapters[Platform.API_SERVER] = adapter
    runner._adapters_for_profile = lambda profile: runner.adapters if profile == 'default' else {}
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    executed = []

    async def execute(authority, ref, row):
        text = row['payload']['text']
        executed.append((ref.session_id, row['principal_id'], text, row['payload'].get('local_automation_v1', {}).get('turn_author')))
        if text == 'fail':
            raise RuntimeError('provider said 429')
        return 'pong'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    try:
        yield SimpleNamespace(authority=authority, chat=chat.session_id, executed=executed,
                              adapter=adapter, home=tmp_path)
    finally:
        adapter._response_store.close()
        adapter._run_idempotency_store.close()
        store.close_all_db_handles()


@pytest_asyncio.fixture
async def peer(peer_runtime):
    adapter = peer_runtime.adapter
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    async with TestClient(TestServer(app)) as cli:
        yield SimpleNamespace(**vars(peer_runtime), cli=cli)


@pytest.mark.asyncio
async def test_peer_dm_into_the_bot_chat_is_the_chats_own_admission(peer):
    resp = await peer.cli.post(f"/api/sessions/{peer.chat}/chat", json={"message": "ping", "author": AUTHOR})
    body = await resp.json()
    assert resp.status == 200, body
    assert body['message']['content'] == 'pong' and body['delivery_id']
    assert peer.executed == [(peer.chat, 'automation:' + peer.authority.sessions[peer.chat].route, 'ping', AUTHOR)]
    # The receipt is durable and exact: the same delivery reads back settled, no re-execution.
    record = json.loads((peer.home / 'runtime' / 'bot_live_delivery' / f"{body['delivery_id']}.json").read_text())
    assert record['status'] == 'settled' and record['session_id'] == peer.chat
    rows = raw_holder(peer.authority.persistence)._read_all("SELECT principal_id FROM session_admissions WHERE principal_id='api'")
    assert rows == [], 'the Bot Chat was never bound as an API conversation'


@pytest.mark.asyncio
async def test_streamed_peer_dm_and_peer_run_ride_the_same_door(peer):
    resp = await peer.cli.post(f"/api/sessions/{peer.chat}/chat/stream", json={"message": "ping", "author": AUTHOR})
    assert resp.status == 200, await resp.text()
    assert resp.content_type == "text/event-stream"
    names = [line[7:] for line in (await resp.text()).splitlines() if line.startswith('event: ')]
    assert names[0] == 'run.started' and 'assistant.completed' in names and names[-1] == 'done'

    resp = await peer.cli.post("/v1/runs", json={"input": "fail", "session_id": peer.chat, "author": AUTHOR})
    run_id = (await resp.json())['run_id']
    for _ in range(100):
        status = await (await peer.cli.get(f"/v1/runs/{run_id}")).json()
        if status['status'] in ('completed', 'failed', 'cancelled'):
            break
        await asyncio.sleep(0.05)
    assert status['status'] == 'failed' and status['delivery_id'], status
    assert [text for _, _, text, _ in peer.executed] == ['ping', 'fail']
    # A turn into any other session still runs as an ordinary API admission.
    resp = await peer.cli.post("/api/sessions/scratch/chat", json={"message": "elsewhere"})
    assert resp.status == 200 and [text for _, _, text, _ in peer.executed][-1] == 'elsewhere'
    assert raw_holder(peer.authority.persistence)._read_all("SELECT 1 FROM session_admissions WHERE principal_id='api'")

async def _submit(peer, text):
    request = make_mocked_request('POST', '/v1/runs', headers={'Idempotency-Key': text})
    request.json = AsyncMock(return_value={'input': text, 'session_id': peer.chat})
    response = await peer.adapter._handle_runs(request)
    assert response.status == 202, response.text
    return json.loads(response.text)['run_id']


async def _run_status(peer, run_id, *, stop=False):
    request = make_mocked_request('POST' if stop else 'GET',
                                  f'/v1/runs/{run_id}' + ('/stop' if stop else ''),
                                  match_info={'run_id': run_id})
    handler = peer.adapter._handle_stop_run if stop else peer.adapter._handle_get_run
    response = await handler(request)
    assert response.status == 200, response.text
    return json.loads(response.text)


async def _wait(event):
    await asyncio.wait_for(event.wait(), 5)


@pytest.mark.asyncio
async def test_peer_stop_reaches_exact_owner_after_observer_loss(peer_runtime, monkeypatch):
    from gateway.session_bot_mailbox import mailbox_lock
    from persistence import AuthorityRefusal

    peer = peer_runtime
    first_action, allow_second_action = asyncio.Event(), asyncio.Event()
    second_action_checked, allow_settlement = asyncio.Event(), asyncio.Event()
    follower_started, finish_follower = asyncio.Event(), asyncio.Event()
    stopped = asyncio.Event()
    stop_commands = []
    partial, late = peer.home / 'partial', peer.home / 'late'

    def control(frame):
        stop_commands.append(frame)
        if len(stop_commands) == 1:
            raise AuthorityRefusal('managed_worker_lost')
        stopped.set()

    async def execute(authority, ref, row):
        text = row['payload']['text']
        if text != 'first':
            peer.executed.append(text)
            follower_started.set()
            await finish_follower.wait()
            return 'follower result'
        authority._managed_workers = {ref.session_id: SimpleNamespace(control=control)}
        try:
            partial.write_text('retained')
            first_action.set()
            await allow_second_action.wait()
            if not stopped.is_set():
                late.write_text('unexpected new action')
            second_action_checked.set()
            await allow_settlement.wait()
            if stopped.is_set():
                authority.pending_results[row['admission_id']] = {
                    'result': {'final_response': 'First action completed.', 'completed': False,
                               'partial': True, 'interrupted': True},
                    'usage': {},
                }
                return 'First action completed.'
            return 'uninterrupted result'
        finally:
            authority._managed_workers.clear()

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    first = await _submit(peer, 'first')
    await _wait(first_action)
    queued = await _submit(peer, 'withdraw')
    follower = await _submit(peer, 'follower')
    try:
        withdrawn = await _run_status(peer, queued, stop=True)
        assert withdrawn['status'] == 'cancelled' and not withdrawn['partial'] and not withdrawn.get('output')
        assert (await _run_status(peer, follower))['status'] == 'queued'
        assert stop_commands == []

        observer = peer.adapter._active_run_tasks[first]
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        peer.adapter._run_statuses.pop(first)
        peer.adapter._run_owners.pop(first)
        binding = await _run_status(peer, first)
        assert binding['status'] == 'running' and binding['execution_generation'] > 0
        assert (await _submit(peer, 'first')) == first
        receipt = peer.home / 'runtime' / 'bot_live_delivery' / (binding['delivery_id'] + '.json')
        unavailable = receipt.with_suffix('.unavailable')
        async with mailbox_lock(peer.authority):
            receipt.rename(unavailable)
        try:
            assert (await _run_status(peer, first, stop=True))['status'] == 'unknown'
            assert stop_commands == []
        finally:
            unavailable.rename(receipt)

        assert (await _run_status(peer, first, stop=True))['status'] == 'stopping'
        assert not stopped.is_set()
        assert (await _run_status(peer, first, stop=True))['status'] == 'stopping'
        assert stopped.is_set()
        assert (await _run_status(peer, first, stop=True))['status'] == 'stopping'
        assert stop_commands == [{'type': 'stop'}, {'type': 'stop'}]

        allow_second_action.set()
        await _wait(second_action_checked)
        assert partial.read_text() == 'retained' and not late.exists()
        assert (await _run_status(peer, first))['status'] == 'stopping'
        assert (await _run_status(peer, follower))['status'] == 'queued'
        allow_settlement.set()
        await _wait(follower_started)
        result = await _run_status(peer, first, stop=True)
        assert result['status'] == 'cancelled' and result['interrupted'] and not result['completed']
        assert result['partial'] and result['output'] == 'First action completed.'
        assert result['admission_id'] == binding['admission_id']
        assert (await _run_status(peer, follower))['status'] == 'running'
        assert peer.executed == ['follower']
        assert len(stop_commands) == 2
    finally:
        allow_second_action.set()
        allow_settlement.set()
        finish_follower.set()
        await asyncio.gather(*list(peer.adapter._active_run_tasks.values()))


@pytest.mark.asyncio
async def test_peer_stop_does_not_cross_execution_generation(peer_runtime, monkeypatch):
    peer = peer_runtime
    first_started, finish_first = asyncio.Event(), asyncio.Event()
    second_started, finish_second = asyncio.Event(), asyncio.Event()
    stop_selected, release_stop = asyncio.Event(), asyncio.Event()
    interrupt = peer.authority.interrupt

    async def delayed_interrupt(actor, ref, generation):
        stop_selected.set()
        await release_stop.wait()
        await interrupt(actor, ref, generation)

    async def execute(authority, ref, row):
        first = row['payload']['text'] == 'old'
        started, finish = (first_started, finish_first) if first else (second_started, finish_second)
        started.set()
        await finish.wait()
        return row['payload']['text'] + ' result'

    monkeypatch.setattr(peer.authority, 'interrupt', delayed_interrupt)
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    old = await _submit(peer, 'old')
    await _wait(first_started)
    new = await _submit(peer, 'new')
    old_generation = peer.authority.executing[peer.chat]
    stop = asyncio.create_task(_run_status(peer, old, stop=True))
    try:
        await _wait(stop_selected)
        finish_first.set()
        await _wait(second_started)
        assert (await _run_status(peer, new))['execution_generation'] > old_generation
        release_stop.set()
        assert (await stop)['status'] == 'completed'
        assert (await _run_status(peer, new))['status'] == 'running'
        assert peer.authority.pending_stops == {}
    finally:
        finish_first.set()
        finish_second.set()
        release_stop.set()
        await asyncio.gather(stop, *list(peer.adapter._active_run_tasks.values()), return_exceptions=True)
    assert (await _run_status(peer, new))['status'] == 'completed'


_PIPE_WORKER = """
import json
import os
from pathlib import Path
import sys

import psutil
from hermes_cli.process_identity import worker_birth

def send(frame):
    print(json.dumps(frame), flush=True)

me = psutil.Process()
send({'type': 'hello', 'pid': me.pid, 'birth': worker_birth(me),
      'ancestors': [os.getppid()]})
assert json.loads(sys.stdin.readline())['version'] == 1
Path(sys.argv[1]).write_text('first action')
if json.loads(sys.stdin.readline()) != {'type': 'stop'}:
    Path(sys.argv[2]).write_text('late action')
    raise SystemExit(1)
send({'type': 'result', 'result': {'final_response': 'Interrupted reply',
      'completed': False, 'interrupted': True, 'partial': True}})
assert json.loads(sys.stdin.readline()) == {'type': 'finish'}
send({'type': 'finished'})
"""


@pytest.mark.asyncio
@pytest.mark.platforms('posix')
@pytest.mark.parametrize('peer_runtime', ['api-peer'], indirect=True)
async def test_recovered_peer_stop_rpc_reaches_managed_worker_pipe(peer_runtime, monkeypatch):
    from gateway import session_managed_worker as managed
    from gateway.owner_identity import host_owner_subject
    from gateway.platforms import api_server_bot_chat
    from gateway.session_controls import AuthorityConnection
    from tools.bot_live_delivery import read_delivery_result

    peer = peer_runtime
    first_action, late_action = peer.home / 'first-action', peer.home / 'late-action'
    children = []

    class PipeWorker(subprocess.Popen):
        def __init__(self, args, **kwargs):
            super().__init__([sys.executable, '-c', _PIPE_WORKER, str(first_action), str(late_action)], **kwargs)
            children.append(self)

    async def execute(authority, ref, row):
        return await managed.execute_managed(authority, ref, row, policy=None)

    monkeypatch.setattr(managed.subprocess, 'Popen', PipeWorker)
    monkeypatch.setattr(managed, '_worker_env', lambda authority: {'HERMES_HOME': str(peer.home)})
    monkeypatch.setattr(managed, '_bootstrap', lambda *args: {'version': 1, 'api_key': ''})
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)

    identity = dict(provider='local', user_id=host_owner_subject(), profile_id=peer.authority.profile_id,
                    instance_id=peer.authority.instance_id, native_bootstrap=True,
                    capabilities=['session:submit', 'session:control'])
    owner = AuthorityConnection(peer.authority, SimpleNamespace(), identity)
    dashboard = AuthorityConnection(peer.authority, SimpleNamespace(),
                                    {'provider': 'basic', 'user_id': 'dashboard'})
    forwarded = AuthorityConnection(peer.authority, SimpleNamespace(),
                                    {**identity, 'user_id': 'auth:v1:["basic","","dashboard"]'})
    loop = asyncio.get_running_loop()

    async def dispatch(connection, params):
        return await connection.dispatch(
            {'jsonrpc': '2.0', 'id': 1, 'method': 'bot_relay.cancel', 'params': params})

    replies = []

    def rpc(home, method, params):
        assert home == peer.home and method == 'bot_relay.cancel'
        reply = asyncio.run_coroutine_threadsafe(dispatch(owner, params), loop).result(timeout=5)
        replies.append(reply)
        return reply['result']

    run_id = await _submit(peer, 'managed-first')
    try:
        await async_wait_until(first_action.exists, timeout=10, what='the child to perform its first action')
        binding = await _run_status(peer, run_id)
        receipt = read_delivery_result(peer.home, binding['delivery_id'])
        assert receipt['principal_id'] == 'api-peer'
        observer = peer.adapter._active_run_tasks[run_id]
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        peer.adapter._run_statuses.pop(run_id)
        scope = peer.adapter._run_owners.pop(run_id)

        relay = dict(id=binding['delivery_id'], profile='default')
        assert (await dispatch(owner, relay))['error']['data']['reason'] == 'permission_denied'
        exact = dict(relay, run_id=run_id, run_scope=scope,
                     admission_id=binding['admission_id'], session_id=binding['session_id'])
        assert (await dispatch(dashboard, exact))['error']['data']['reason'] == 'permission_denied'
        assert (await dispatch(forwarded, exact))['error']['data']['reason'] == 'permission_denied'
        assert (await dispatch(owner, {**exact, 'run_scope': 'other'}))['error']['data']['reason'] == 'permission_denied'
        wrong_admission = await dispatch(owner, {**exact, 'admission_id': 'other'})
        assert wrong_admission['error']['data']['reason'] == 'admission_conflict'
        assert children[0].poll() is None

        monkeypatch.setattr(api_server_bot_chat, '_authority', lambda runner: None)
        monkeypatch.setattr('tools.bot_live_delivery._authority_rpc', rpc)
        await _run_status(peer, run_id, stop=True)
        assert replies[0]['result']['status'] in ('stopping', 'cancelled')
        await async_wait_until(
            lambda: read_delivery_result(peer.home, binding['delivery_id'])['status'] == 'cancelled',
            timeout=10, what='the owner to persist the interrupted receipt')
        result = await _run_status(peer, run_id)
        assert result['status'] == 'cancelled' and result['interrupted'] and not result['completed']
        assert result['partial'] and result['output'] == 'Interrupted reply'
        assert first_action.read_text() == 'first action' and not late_action.exists()
        assert children[0].returncode == 0
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
        await asyncio.gather(peer.authority.sessions[peer.chat].task, return_exceptions=True)
