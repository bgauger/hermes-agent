"""Who may drive a profile's Bot Desktop screen: the agent (default) or exactly one human viewer.

The lease is the single truth shared by the RFB bridge (drops human input from non-holders), the
``computer_use`` tool (refuses to act while a human holds control — the person may be typing a
credential, so even screenshots are refused; fail closed rather than trusting the agent to pause
itself) and the Desktop UI (Watch / Take over / Hand back).

Scope: the lease is a TOOL-LEVEL fence, not a property of the X server. ``computer_use`` and the browser
tools consult it; a process the agent starts by hand against the published ``DISPLAY``/``XAUTHORITY``
(the ``terminal`` tool, a script) is inside the documented same-user boundary and is not stopped by it
(bot-screen.md, "Threat model"; #110040).

Authority lives ON DISK, ``<HERMES_HOME>/bot-desktop/lease.json`` under an fcntl lock, because the
processes that must agree do not share memory: ``hermes serve`` (viewer bridge), the messaging
gateway, a CLI turn and isolated workers all drive the same display. Every read goes to the file;
the in-process Condition only wakes local waiters early. ``epoch`` increments on every transition so
an action admitted under one lease can tell that control changed underneath it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional
from uuid import uuid4

from hermes_constants import get_hermes_home, hermes_home_key, secure_parent_dir

try:
    import fcntl
except ImportError:  # Windows/macOS without fcntl: computer_use imports this module on every call, and no
    fcntl = None     # multi-process Bot Desktop exists there, so the cross-process lock degrades to a no-op.

logger = logging.getLogger(__name__)

AGENT = "agent"
HUMAN = "human"


class HumanHasControl(RuntimeError):
    """Raised by screen-driving tools while a human holds the lease."""


@dataclass
class Lease:
    holder: str = AGENT
    viewer_id: Optional[str] = None
    since: float = field(default_factory=time.time)
    reason: str = ""
    epoch: int = 0
    explicit_release: bool = False
    pending: bool = False
    input_unconfirmed: bool = False
    handback_pending: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def public_view(lease: Lease) -> dict[str, object]:
    """The lease as anything outside the gateway may see it (RPC results, the ``display.lease`` broadcast,
    the CLI): the holder's viewer id is a capability — whoever presents it co-drives or releases the lease —
    so it is replaced by a short hash the holder can match against its own id to know it is in control."""
    d = lease.as_dict()
    del d["pending"], d["explicit_release"], d["input_unconfirmed"], d["handback_pending"]
    d["viewer_id"] = None
    d["viewer_hash"] = (
        hashlib.sha256(lease.viewer_id.encode()).hexdigest()[:12]
        if lease.viewer_id and not lease.pending else None
    )
    return d


_lock = threading.Condition()
_listeners: list[Callable[[str, Lease], None]] = []


def _path(profile_key: Optional[str]) -> Path:
    """``profile_key`` is the HERMES_HOME path of the profile whose lease is meant (the RFB bridge
    serves several profiles from one process); ``None`` means the current profile."""
    home = Path(profile_key) if profile_key else get_hermes_home()
    return home / "bot-desktop" / "lease.json"


def _read(path: Path) -> Lease:
    """No file = fresh profile, agent holds. A file that exists but cannot be parsed is a torn write
    or tampering: fail CLOSED (human holds) — an unreadable lease must never let the agent act on a
    screen a human may be using; the next successful write repairs it."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return Lease()
    except (OSError, ValueError):
        data = None
    if isinstance(data, dict):
        lease = Lease(**{k: v for k, v in data.items() if k in Lease.__dataclass_fields__})
        if (lease.holder in (AGENT, HUMAN) and "holder" in data
                and type(lease.epoch) is int and lease.epoch >= 0
                and type(lease.explicit_release) is bool and type(lease.pending) is bool
                and type(lease.input_unconfirmed) is bool and type(lease.handback_pending) is bool
                and (not lease.handback_pending or lease.pending and lease.input_unconfirmed)
                and (not lease.input_unconfirmed or lease.holder == HUMAN and lease.explicit_release)
                and isinstance(lease.since, (int, float)) and isinstance(lease.reason, str)
                and (isinstance(lease.viewer_id, str) or lease.viewer_id is None)
                and (lease.holder == HUMAN or lease.viewer_id is None and not lease.pending)):
            if lease.holder == HUMAN and lease.explicit_release:
                # Older leases carry no evidence about input already forwarded by their bridge.
                if "input_unconfirmed" not in data:
                    lease.input_unconfirmed = True
                if "pending" not in data:
                    lease.pending = True
            return lease
    try:
        epoch = path.stat().st_mtime_ns // 1000
    except OSError:
        epoch = time.time_ns() // 1000
    return Lease(holder=HUMAN, reason="lease file unreadable or corrupt", pending=True, epoch=epoch)


def _private_dir(path: Path) -> None:
    """``bot-desktop/`` owner-only even when the lease is the first thing written there (a takeover can be
    recorded before start() ever ran, and the umask would otherwise leave it 0755)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    secure_parent_dir(path)


def _open_private(path: str | bytes | os.PathLike, flags: int) -> int:
    """``open(..., opener=_open_private)``: the file is created 0600 regardless of the umask."""
    return os.open(path, flags, 0o600)


def _write(path: Path, lease: Lease) -> None:
    _private_dir(path)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8", opener=_open_private) as fh:
        fh.write(json.dumps(lease.as_dict()))
    os.replace(tmp, path)


class _locked:
    """Cross-process critical section over the lease file (fcntl on a sibling lock file)."""

    def __init__(self, path: Path):
        self._lockfile = path.with_suffix(".lock")
        self._fh = None

    def __enter__(self):
        if fcntl is None:
            return self
        _private_dir(self._lockfile)
        self._fh = open(self._lockfile, "a+", encoding="utf-8", opener=_open_private)
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self._fh is None:
            return
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        self._fh.close()


def get(profile_key: Optional[str] = None) -> Lease:
    return _read(_path(profile_key))


def on_change(listener: Callable[[str, Lease], None]) -> Callable[[], None]:
    """Subscribe to lease transitions made IN THIS PROCESS (the gateway broadcasts them to Desktop
    clients). Transitions made by another process are observed by reading, not by callback."""
    with _lock:
        _listeners.append(listener)

    def _off() -> None:
        with _lock:
            if listener in _listeners:
                _listeners.remove(listener)
    return _off


def _notify(key: str, lease: Lease) -> None:
    # Snapshot under the lock: a viewer disconnecting on another thread runs unsubscribe()
    # concurrently with this loop; removing an entry from the live list mid-iteration skips
    # the next listener, and a still-connected viewer misses the lease transition.
    with _lock:
        listeners = list(_listeners)
    for cb in listeners:
        try:
            cb(key, lease)
        except Exception:  # a broken subscriber must not wedge the handoff
            pass


def _transition(profile_key: Optional[str], mutate: Callable[[Lease], bool]) -> Lease:
    key, path = hermes_home_key(profile_key) if profile_key else hermes_home_key(), _path(profile_key)
    with _locked(path):
        lease = _read(path)
        if not mutate(lease):
            return lease
        lease.epoch += 1
        _write(path, lease)
    with _lock:
        _lock.notify_all()
    _notify(key, lease)
    return lease


def explicit_release_enabled(profile_key: Optional[str] = None) -> bool:
    from agent.runtime_scope import bind_state, scope_for_profile
    from hermes_cli.config import load_config_readonly

    if profile_key:
        with bind_state(scope_for_profile(profile_key), secrets={}, terminal=False):
            config = load_config_readonly()
    else:
        config = load_config_readonly()
    settings = config.get("bot_desktop")
    return isinstance(settings, dict) and settings.get("explicit_release") is True


def explicit_release_applies(profile_key: Optional[str] = None) -> bool:
    """Keep the policy of an active lease even after a configuration change."""
    return get(profile_key).explicit_release or explicit_release_enabled(profile_key)


def _drained(path: Path) -> bool:
    try:
        return next((path.parent / "actions").iterdir(), None) is None
    except FileNotFoundError:
        return True
    except OSError:
        return False


def acquire(viewer_id: str, *, profile_key: Optional[str] = None, reason: str = "") -> Lease:
    """Close agent admission before acknowledging human control."""
    configured = explicit_release_enabled(profile_key)
    path = _path(profile_key)

    def _m(lease: Lease) -> bool:
        drained = _drained(path)
        explicit = lease.explicit_release or configured or not drained
        # Until the RFB executor confirms prior input, another viewer cannot start driving either.
        if lease.input_unconfirmed and lease.viewer_id != viewer_id:
            return False
        pending = explicit and (lease.pending or fcntl is None or not drained)
        if lease.holder == HUMAN and lease.viewer_id == viewer_id:
            changed = (lease.explicit_release, lease.pending) != (explicit, pending)
            lease.explicit_release, lease.pending = explicit, pending
            return changed
        # The agent's ask ("please log in to X") stays as the takeover reason: the human needs it
        # on screen WHILE they act, not only before they clicked Take over.
        lease.holder, lease.viewer_id, lease.since = HUMAN, viewer_id, time.time()
        lease.reason = reason or ""
        lease.explicit_release, lease.pending = explicit, pending
        return True
    return _transition(profile_key, _m)


def release(viewer_id: Optional[str] = None, *, profile_key: Optional[str] = None,
            unless_human: bool = False, epoch: Optional[int] = None, implicit: bool = False) -> Lease:
    """Return control to the agent. With ``viewer_id`` only that holder may release (a stale viewer
    closing its window must not yank control from the one who took over after it). ``unless_human``
    makes a bare release a no-op while any human holds: the decision is taken under the file lock,
    so a takeover racing a check-then-release cannot be silently revoked. ``epoch`` releases only the
    lease generation the caller judged: any transition since (a takeover, a hand-back and re-take by the
    same viewer) keeps the lease. Callers read the returned lease's holder to learn whether anything
    happened. Implicit release from idle or viewer teardown respects the policy saved with the lease."""
    def _m(lease: Lease) -> bool:
        if epoch is not None and lease.epoch != epoch:
            logger.info("bot-desktop lease: release of epoch %d ignored, the lease changed since", epoch)
            return False
        if unless_human and lease.holder == HUMAN:
            logger.info("bot-desktop lease: bare release ignored, a human holds")
            return False
        if implicit and lease.explicit_release:
            return False
        if viewer_id is not None and lease.holder == HUMAN and lease.viewer_id != viewer_id:
            # Ignored, not an error: the returned lease still shows the real holder. Logged so a
            # caller that never inspects the return value leaves a trace.
            logger.info("bot-desktop lease: release by %r ignored, another viewer holds", viewer_id)
            return False
        if lease.holder == AGENT:
            # Already the agent's. Bumping the epoch here would make a legitimately admitted in-flight
            # agent action (a double-clicked Hand back, a stray CLI stop) look overtaken and get voided.
            return False
        if lease.input_unconfirmed:
            if lease.handback_pending:
                return False
            lease.handback_pending = lease.pending = True
            return True
        lease.holder, lease.viewer_id, lease.since, lease.reason = AGENT, None, time.time(), ""
        lease.explicit_release = lease.pending = False
        return True
    return _transition(profile_key, _m)


def human_holds(profile_key: Optional[str] = None) -> bool:
    return get(profile_key).holder == HUMAN


def viewer_may_send_input(viewer_id: str, *, profile_key: Optional[str] = None) -> bool:
    return permits_input(get(profile_key), viewer_id)


def permits_input(lease: Lease, viewer_id: str) -> bool:
    return lease.holder == HUMAN and not lease.pending and lease.viewer_id == viewer_id


@contextmanager
def input_snapshot(profile_key: str) -> Iterator[Lease]:
    path = _path(profile_key)
    with _locked(path):
        yield _read(path)


def record_unconfirmed_input(profile_key: str, snapshot: Lease) -> None:
    """Persist the uncertainty before enqueueing input, under the caller's input_snapshot lock."""
    if snapshot.explicit_release and not snapshot.input_unconfirmed:
        snapshot.input_unconfirmed = True
        _write(_path(profile_key), snapshot)


def assert_agent_may_act(profile_key: Optional[str] = None) -> Lease:
    """The lease as of now, or ``HumanHasControl``. Callers keep the returned ``epoch`` and compare it
    with ``get().epoch`` after an admitted action: a change means a human took over mid-flight."""
    lease = get(profile_key)
    if lease.holder == HUMAN:
        raise HumanHasControl(
            "A human has taken over this desktop (they may be entering a credential). Screen actions and "
            "captures are refused until they hand control back. Tell the user what you need in your reply.")
    return lease


@dataclass
class _Action:
    uncertain: bool = False


_current_action: ContextVar[Optional[_Action]] = ContextVar("bot_desktop_action", default=None)


def executor_outcome_unknown() -> None:
    """A client stopped waiting without a terminal executor response."""
    action = _current_action.get()
    if action is not None:
        action.uncertain = True


@contextmanager
def admitted_action(profile_key: Optional[str] = None) -> Iterator[Lease]:
    """Track an action until its executor settles, even if the caller process exits."""
    if not explicit_release_enabled(profile_key):
        yield assert_agent_may_act(profile_key)
        return

    path = _path(profile_key)
    if fcntl is None:
        raise HumanHasControl("Bot Desktop cannot verify cross-process action admission on this host.")
    ticket = path.parent / "actions" / uuid4().hex
    with _locked(path):
        admitted = assert_agent_may_act(profile_key)
        ticket.parent.mkdir(exist_ok=True)
        os.close(os.open(ticket, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))

    action = _Action()
    token = _current_action.set(action)
    try:
        yield admitted
        if not action.uncertain:
            def settle(lease: Lease) -> bool:
                ticket.unlink()
                if (lease.pending and not lease.handback_pending and lease.explicit_release
                        and lease.viewer_id and _drained(path)):
                    lease.pending = False
                    return True
                return False

            _transition(str(path.parent.parent), settle)
    finally:
        _current_action.reset(token)
