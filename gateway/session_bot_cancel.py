"""``bot_relay.cancel``: a ``message_agent`` sender withdraws its own delivery by ``delivery_id``.

The cancel runs on the delivery's owner under the same mailbox lock that admits the delivery's one
automatic retry (``gateway/session_bot.py::_maybe_retry``), so it always sees which admission is the
request now. A sender that picked an admission from the receipt file could race that retry: the
first attempt already failed, the retry not yet recorded, and "nothing to cancel" while it runs.

Whatever this finds, a retry not admitted yet is withdrawn for good (the receipt's ``retry`` marks
it refused ``cancelled``), so no later wake, re-delivery or owner restart admits it.
"""
import asyncio

from gateway.session_contract import SessionActor, SessionRef
from gateway.session_persistence import authority_persistence
from persistence.core.errors import AuthorityRefusal

_FIELDS = frozenset({'id', 'profile', 'author_id', 'admission_id', 'session_id'})


async def cancel(connection, params, *, peer=False):
    from gateway.session_authorities import require_serving
    from gateway.session_runtime_workers import track_mutation
    require_serving(connection.authority)
    return await asyncio.shield(track_mutation(connection.authority, _cancel(connection, params, peer=peer)))


async def _cancel(connection, params, *, peer):
    from gateway.session_bot import (_delivery_id, _delivery_result, _home, _retry_admission, _retry_eligible,
                                     _retry_identity)
    from gateway.session_bot_mailbox import mailbox_lock, read_receipt, remember_receipt, write_receipt
    authority, actor = connection.authority, connection.actor
    if set(params) - _FIELDS:
        raise AuthorityRefusal('invalid_params')
    home = (authority_persistence(authority).scope.profile_dir if peer
            else _home(authority, actor, params.get('profile')))
    try:
        key = _delivery_id(params.get('id'))
    except ValueError as exc:
        raise AuthorityRefusal('invalid_params') from exc
    async with mailbox_lock(authority):
        _, record = await read_receipt(home, key)
        if record is None or record.get('delivery_id') != key or not record.get('admission_id'):
            raise AuthorityRefusal('not_found')
        # Only the delivery's sender can use the relay RPC. The HTTP edge has already
        # authenticated its run and can act on that run's original mailbox sender.
        if not peer and (record['principal_id'] != actor.subject
                         or (record.get('author') or {}).get('id') != params.get('author_id')):
            raise AuthorityRefusal('permission_denied')
        if peer or 'admission_id' in params or 'session_id' in params:
            if (params.get('admission_id'), params.get('session_id')) != (record['admission_id'], record['session_id']):
                raise AuthorityRefusal('admission_conflict')
            return await _stop_bound_delivery(
                authority, _peer_actor(authority, record) if peer else actor, home, record)
        retry_id = _retry_admission(authority, record)
        current = retry_id or record['admission_id']
        row = authority_persistence(authority).admissions.get(admission_id=current)
        live = row is not None and row.status != 'terminal'
        withdrawn = retry_id is None and (live or _retry_eligible(authority, record))
        if withdrawn:
            record['retry'] = {**(record.get('retry') or {}), 'identity': _retry_identity(key), 'refused': 'cancelled'}
        status = 'finished'
        if live:
            answer = await connection.cancel(SessionRef(authority.profile_id, record['session_id']),
                                             {'admission_id': current})
            status = 'stopping'  # a running turn was told to stop; its receipt reports how it ended
            if answer['status'] == 'terminal':
                status = 'cancelled' if answer.get('outcome') == 'cancelled' else 'finished'
        elif withdrawn:
            status = 'cancelled'  # its first attempt failed; the retry it was owed never starts
        record.update(_delivery_result(authority, record))
        await write_receipt(home, record)
        remember_receipt(authority, home, record)
        return {'delivery_id': key, 'status': status, 'receipt_status': record['status']}


def _peer_actor(authority, record):
    capabilities = {'session:read', 'session:submit', 'session:control'}
    if record.get('operator') is True:
        capabilities.add('session:operator')
    return SessionActor(record['principal_id'], authority.profile_id, frozenset(capabilities), 'api-peer-stop')


async def _stop_bound_delivery(authority, actor, home, record):
    from gateway.session_bot import _delivery_result, _retry_admission
    from gateway.session_bot_mailbox import remember_receipt, write_receipt

    store = authority_persistence(authority)
    retry_id = _retry_admission(authority, record)
    admission_id = retry_id or record['admission_id']
    row = store.admissions.get(admission_id=admission_id)
    ref = SessionRef(authority.profile_id, record['session_id'])
    authority.authorize(actor, ref, 'session:control')
    if row is None or row.target_session_id != ref.session_id:
        raise AuthorityRefusal('unknown_execution')

    if not retry_id and (record.get('retry') or {}).get('refused') != 'peer_stop':
        # Commit withdrawal before signalling the worker: even if the acknowledgement is lost,
        # this delivery must never acquire another execution after Stop.
        record['retry'] = {**(record.get('retry') or {}), 'refused': 'peer_stop'}
        await write_receipt(home, record)
    try:
        await _stop_claim(authority, actor, ref, row, record)
    except AuthorityRefusal as exc:
        if exc.reason != 'stale_generation':
            raise
        current = store.admissions.get(admission_id=admission_id)
        if current is None or current.status != 'terminal':
            raise
    record.update(_delivery_result(authority, record))
    await write_receipt(home, record)
    remember_receipt(authority, home, record)
    return {'delivery_id': record['delivery_id'], 'receipt_status': record['status']}


async def _stop_claim(authority, actor, ref, row, record):
    from gateway.session_managed_worker import interrupt_managed

    if row.status == 'queued':
        claimed = await authority.cancel_or_claimed(actor, ref, row.admission_id)
        if claimed is None:
            return
        generation = claimed['generation']
    elif row.status == 'started':
        generation = row.generation
    else:
        return
    if record.get('peer_stop_generation') == generation:
        return
    if not interrupt_managed(authority, actor, ref, generation):
        await authority.interrupt(actor, ref, generation)
    record['peer_stop_generation'] = generation
