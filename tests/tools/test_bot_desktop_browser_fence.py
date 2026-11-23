"""The bot's browser tools obey the screen lease: while a human holds the shared browser, nothing is
dispatched, and a command whose run crossed a takeover loses its result."""

from __future__ import annotations

import json
import subprocess

import pytest

from tools.bot_desktop import lease, runtime
from tests.bot_desktop_lease import reset_lease


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    reset_lease()
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":37"})
    yield
    reset_lease()


def _wire_browser_exec(monkeypatch, run_cli):
    """Route browser_exec through the local session layer without starting a real browser."""
    from tools import browser_lightpanda, browser_session, browser_session_cdp, browser_session_cloud
    from tools import browser_use_cli as browser_use

    monkeypatch.setattr(browser_use, "_find_cli", lambda: ["browser-use"])
    monkeypatch.setattr(browser_use, "_base_subprocess_env", dict)
    monkeypatch.setattr(browser_use, "_real_profile_consented", lambda: False)
    monkeypatch.setattr(browser_lightpanda, "using_lightpanda_engine", lambda: False)
    monkeypatch.setattr(browser_session_cdp, "get_cdp_override", lambda: "")
    monkeypatch.setattr(browser_session_cdp, "get_cdp_override_raw", lambda: "")
    monkeypatch.setattr(browser_session_cloud, "get_cloud_provider", lambda: None)
    monkeypatch.setattr(browser_session, "get_session_info", lambda *_a, **_kw: {
        "session_name": "review", "cdp_url": "http://127.0.0.1:9222", "features": {"local": True}})
    monkeypatch.setattr(browser_use, "_attach_vault_supervisor", lambda *a: None)
    monkeypatch.setattr(browser_use, "_run_cli_killing_process_group", run_cli)
    return browser_use


def test_browser_exec_is_fenced_while_human_controls_shared_browser(monkeypatch):
    dispatched: list[str] = []

    def run_cli(*_args):
        dispatched.append("browser-use")
        return subprocess.CompletedProcess([], 0, "WHAT-THE-HUMAN-TYPED", "")

    browser_use = _wire_browser_exec(monkeypatch, run_cli)
    lease.acquire("human-viewer")
    raw = browser_use.browser_exec("print(page_info())", task_id="review")
    assert isinstance(raw, str)
    result = json.loads(raw)
    assert dispatched == [], "human holds the lease, yet browser_exec was dispatched"
    assert "WHAT-THE-HUMAN-TYPED" not in raw
    assert result.get("code") == "human_has_control"


def test_browser_exec_result_crossing_a_takeover_is_discarded(monkeypatch):
    def run_cli(*_args):
        lease.acquire("human-viewer")
        lease.release("human-viewer")
        return subprocess.CompletedProcess([], 0, "WHAT-THE-HUMAN-TYPED", "")

    browser_use = _wire_browser_exec(monkeypatch, run_cli)
    raw = browser_use.browser_exec("print(page_info())", task_id="review")
    assert isinstance(raw, str)
    result = json.loads(raw)
    assert "WHAT-THE-HUMAN-TYPED" not in raw
    assert result.get("code") == "human_has_control"


def test_run_fenced_refuses_a_local_session_while_human_controls_shared_browser():
    from tools.browser_session import run_fenced

    dispatched: list = []
    lease.acquire("human-viewer")
    result = run_fenced({"session_name": "review", "cdp_url": None, "features": {"local": True}},
                        lambda: dispatched.append("cmd") or {"success": True})
    assert dispatched == [], f"human holds the lease, yet a browser command was dispatched: {dispatched}"
    assert result.get("code") == "human_has_control"


def test_run_fenced_discards_a_result_crossing_a_takeover():
    from tools.browser_session import run_fenced

    def run_then_takeover():
        lease.acquire("human-viewer")
        lease.release("human-viewer")  # a full cycle, control is back — the frame is still theirs
        return {"success": True, "data": {"secret": "WHAT-THE-HUMAN-TYPED"}}

    result = run_fenced({"session_name": "review", "cdp_url": None, "features": {"local": True}},
                        run_then_takeover)
    assert "WHAT-THE-HUMAN-TYPED" not in json.dumps(result)


def test_real_profile_local_browser_is_fenced_by_provenance_even_without_a_live_display(monkeypatch):
    """A real-profile session attaches over a loopback cdp_url but is launched with the Bot Desktop
    DISPLAY, so it IS the human's browser: the fence keys on the ``local`` feature, not on the
    transport. And a stranded human lease with the screen already down must still fence (computer_use
    does), not silently unfence the browser."""
    from tools.browser_session import run_fenced

    dispatched: list = []
    monkeypatch.setattr(runtime, "published_env", dict)
    lease.acquire("human-viewer")
    result = run_fenced({"session_name": "rp_1", "cdp_url": "ws://127.0.0.1:9222/devtools/browser/x",
                         "features": {"local": True, "real_profile": True}},
                        lambda: dispatched.append("cmd") or {"success": True})
    assert dispatched == [], f"human holds the lease, yet a real-profile browser command was dispatched: {dispatched}"
    assert result.get("code") == "human_has_control"


def test_vault_page_operations_respect_the_human_lease(monkeypatch):
    """The vault tools reach the page over the supervisor socket, outside `_run_browser_command`: while a
    human holds the screen they must be refused like every other page access, and never focus, inspect or
    write to the form the human is typing into."""
    from tools import browser_session
    from tools import browser_vault_tool as vault

    monkeypatch.setitem(browser_session._active_sessions, "default",
                        {"session_name": "review", "cdp_url": None, "features": {"local": True}})
    touched = []
    for name in ("browser_vault_fill", "browser_vault_enter_code", "browser_vault_save_login"):
        monkeypatch.setattr(vault, name, lambda *a, **k: touched.append(name) or json.dumps({"success": True}))
    lease.acquire("human")
    for action in ("fill", "enter_code", "save_login"):
        res = json.loads(vault._handle_vault({"action": action, "handle": "vault_x"}, task_id="default"))
        assert res["code"] == "human_has_control", action
    assert touched == [], "no vault page access while the human holds the screen"
    lease.release("human")
    assert json.loads(vault._handle_vault({"action": "fill", "handle": "vault_x"}, task_id="default"))["success"] is True


def test_secret_write_re_admits_after_a_takeover_during_the_code_prompt(monkeypatch):
    """`enter_code` blocks on the user's code INSIDE the handler fence. A takeover during that wait must
    refuse the write itself: the outer fence only discards the result afterwards, and by then the code is
    already in the page the human is typing into."""
    from tools import browser_session
    from tools import browser_vault_tool as vault

    monkeypatch.setitem(browser_session._active_sessions, "default",
                        {"session_name": "review", "cdp_url": None, "features": {"local": True}})
    evaluated = []

    class Sup:
        def evaluate_runtime(self, expr):
            evaluated.append(expr)
            return {"ok": True, "result": "{}"}

    monkeypatch.setattr(vault, "_ensure_supervisor", lambda tid: Sup())
    assert vault._eval_js_secret("default", "fill()")["success"] is True  # agent holds: writes
    lease.acquire("human")  # takeover while the prompt was open
    res = vault._eval_js_secret("default", "fill()")
    assert res["success"] is False and res["error_type"] == "human_has_control"
    assert evaluated == ["fill()"], "the credential must not reach the page under a human lease"


@pytest.mark.platforms("posix")
def test_client_timeout_does_not_settle_the_harness_daemon(monkeypatch):
    from pathlib import Path
    from hermes_constants import get_hermes_home

    Path(get_hermes_home(), "config.yaml").write_text("bot_desktop:\n  explicit_release: true\n")
    calls = []

    def run_cli(*args):
        calls.append("daemon still running")
        raise subprocess.TimeoutExpired("fake browser-harness", 1)

    browser_use = _wire_browser_exec(monkeypatch, run_cli)
    result = browser_use.browser_exec("print(page_info())", task_id="review")
    assert "daemon may still be working" in result
    assert lease.acquire("viewer").pending
    assert not lease.viewer_may_send_input("viewer")
    assert json.loads(browser_use.browser_exec("print(page_info())", task_id="review"))["code"] == "human_has_control"
    assert calls == ["daemon still running"]


@pytest.fixture
def explicit_policy():
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "config.yaml").write_text("bot_desktop:\n  explicit_release: true\n")


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("route", ["startup", "replacement", "takeover_during_replacement",
                                   "configured_cdp", "operator_cdp", "sandbox"])
def test_browser_route_setup_is_inside_exclusive_admission(monkeypatch, explicit_policy, route):
    from tools import browser_session, browser_session_cdp, browser_session_lifecycle, browser_session_sandbox

    effects = []

    def run_cli(*args):
        effects.append("dispatch")
        return subprocess.CompletedProcess([], 0, "done", "")

    get_session_info = browser_session.get_session_info
    browser_use = _wire_browser_exec(monkeypatch, run_cli)
    key = browser_use._backend_cache_key("route-test")
    monkeypatch.setattr(browser_session, "get_session_info", get_session_info)
    monkeypatch.setattr(browser_session_lifecycle, "start_cleanup_thread", lambda: None)
    for attr in ("_active_sessions", "_session_last_activity", "_session_owner_homes"):
        monkeypatch.setattr(browser_session, attr, {})

    endpoint = "ws://127.0.0.1:12345/devtools/browser/fake"

    def create(*args, **kwargs):
        effects.append("create")
        # Startup itself is tracked. The viewer cannot drive until routing and dispatch return.
        assert lease.acquire("viewer").pending
        assert not lease.viewer_may_send_input("viewer")
        return {"features": {"local": True}, "cdp_url": endpoint}

    if route != "sandbox":
        monkeypatch.setattr(browser_session, "_create_local_session", create)
    if route in {"replacement", "takeover_during_replacement"}:
        browser_session._active_sessions[key] = {"expires_at": 0, "features": {"local": True}}
        monkeypatch.setattr(browser_session_cdp, "stop_cdp_supervisor", lambda *_: effects.append("evict"))
    if route == "configured_cdp":
        def get_override():
            effects.append("discover")
            return endpoint
        monkeypatch.setattr(browser_session_cdp, "get_cdp_override", get_override)
    if route == "operator_cdp":
        monkeypatch.setattr(browser_use, "_base_subprocess_env", lambda: {"BU_CDP_WS": endpoint})
    if route == "sandbox":
        from tools.bot_desktop import placement
        monkeypatch.setattr(runtime, "tool_placement", lambda: placement.TERMINAL)
        monkeypatch.setattr(browser_session_sandbox, "create_sandbox_session", create)

    lease.acquire("viewer")
    result = json.loads(browser_use.browser_exec("print(1)", task_id="route-test"))
    assert result["code"] == "human_has_control"
    assert effects == []
    if route in {"replacement", "takeover_during_replacement"}:
        assert browser_session._active_sessions[key]["expires_at"] == 0

    lease.release("viewer")
    if route == "takeover_during_replacement":
        def expire(info):
            assert lease.acquire("viewer").pending
            return True
        monkeypatch.setattr(browser_session_lifecycle, "session_has_expired", expire)
        assert json.loads(browser_use.browser_exec("print(1)", task_id="route-test"))["code"] == "human_has_control"
        assert effects == []
        assert lease.viewer_may_send_input("viewer")
        lease.release("viewer")
        return
    result = json.loads(browser_use.browser_exec("print(1)", task_id="route-test"))
    assert effects[-1] == "dispatch"
    if route in {"startup", "replacement", "sandbox"}:
        assert "create" in effects
        assert result["code"] == "human_has_control"
        assert lease.viewer_may_send_input("viewer")
    else:
        assert result["success"] is True
    lease.release("viewer")


def test_cdp_dialog_and_supervisor_routes_refuse_only_the_exclusive_profile(monkeypatch, tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from tools import browser_cdp_tool, browser_dialog_tool, browser_session_cdp

    touched = []
    endpoint = "ws://127.0.0.1:12345/devtools/browser/fake"
    monkeypatch.setattr(browser_cdp_tool, "_WS_AVAILABLE", True)
    monkeypatch.setattr(browser_cdp_tool, "_resolve_cdp_endpoint", lambda: touched.append("raw") or "")
    monkeypatch.setattr(browser_cdp_tool, "_browser_cdp_private_guard", lambda **kw: None)
    monkeypatch.setattr(browser_cdp_tool, "_browser_cdp_via_supervisor", lambda **kw: touched.append("frame"))
    monkeypatch.setattr(browser_dialog_tool.SUPERVISOR_REGISTRY, "get", lambda *a: touched.append("dialog"))
    monkeypatch.setattr(browser_session_cdp, "get_cdp_override", lambda: touched.append("attach") or endpoint)
    monkeypatch.setattr(browser_dialog_tool.SUPERVISOR_REGISTRY, "get_or_start", lambda **kw: None)

    exclusive = tmp_path / "exclusive"
    exclusive.mkdir()
    (exclusive / "config.yaml").write_text("bot_desktop:\n  explicit_release: true\n")
    token = set_hermes_home_override(exclusive)
    try:
        lease.acquire("viewer")
        assert json.loads(browser_cdp_tool.browser_cdp("Target.getTargets"))["code"] == "exclusive_desktop_route"
        assert json.loads(browser_cdp_tool.browser_cdp("Target.getTargets", frame_id="frame"))["code"] == "exclusive_desktop_route"
        assert json.loads(browser_dialog_tool.browser_dialog("dismiss"))["code"] == "exclusive_desktop_route"
        browser_session_cdp.ensure_cdp_supervisor("test")
        assert touched == []
    finally:
        reset_hermes_home_override(token)

    browser_cdp_tool.browser_cdp("Target.getTargets")
    browser_cdp_tool.browser_cdp("Target.getTargets", frame_id="frame")
    browser_dialog_tool.browser_dialog("dismiss")
    browser_session_cdp.ensure_cdp_supervisor("test")
    assert touched == ["raw", "raw", "frame", "dialog", "attach"]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("explicit", [True, False])
def test_worker_exit_and_next_worker_reap_respect_held_browser(tmp_path, explicit):
    import os
    import sys
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    if explicit:
        (home / "config.yaml").write_text("bot_desktop:\n  explicit_release: true\n")
    lease.acquire("viewer")
    # The child exits normally, triggering the actual browser atexit path. Substitute only process
    # identity and effects at the Chromium/harness boundaries.
    worker = r"""
import os
import sys
from pathlib import Path
from tools import browser_launcher as launcher, browser_session, browser_use_cli as harness

root = Path(sys.argv[1])
launcher._shared_profiles_root = lambda: root / "records"
launcher._is_recorded_browser = lambda *a: True

def effect(name):
    with (root / "effects").open("a") as out:
        print(name, file=out)

launcher._terminate = lambda *a: effect("chromium")
harness._reload_daemons = lambda daemons: effect("harness") if daemons else None

if sys.argv[2] == "exit":
    browser = launcher._make_browser("key", 12345, "/devtools/browser/fake", pid=os.getpid(),
                                     start_time=None, user_data_dir=str(root / "ud"), owned=True,
                                     ephemeral=False, home_key=os.environ["HERMES_HOME"])
    launcher._browsers["key"] = browser
    launcher._write_record(browser)
    harness._driven_daemons[("dummy",)] = os.environ["HERMES_HOME"]
else:
    launcher.reap_orphaned_browsers()
"""
    env = {**os.environ, "HERMES_HOME": str(home)}
    for phase in ("exit", "restart"):
        subprocess.run([sys.executable, "-c", worker, str(tmp_path), phase], check=True, timeout=20, env=env)
    effects = tmp_path / "effects"
    if explicit:
        assert not effects.exists()
        assert list((tmp_path / "records").rglob("*.json"))
        assert lease.viewer_may_send_input("viewer")
        lease.release("viewer")
        subprocess.run([sys.executable, "-c", worker, str(tmp_path), "restart"],
                       check=True, timeout=20, env=env)
        assert effects.read_text().splitlines() == ["chromium"]
    else:
        assert set(effects.read_text().splitlines()) == {"chromium", "harness"}


@pytest.mark.platforms("posix")
def test_sandbox_session_can_reattach_after_explicit_worker_cleanup(monkeypatch, tmp_path, explicit_policy):
    from tools import browser_session, browser_session_lifecycle, browser_session_sandbox as sandbox
    from tools.bot_desktop import sandbox_host
    from tools.environments import streams

    effects = []
    env = object()
    udd = str(tmp_path / "fake-browser-profile")
    found = iter((sandbox.GONE, (12345, "/devtools/browser/fake")))
    monkeypatch.setattr(sandbox, "_sandbox", lambda: (env, {"DISPLAY": ":37"}))
    monkeypatch.setattr(sandbox_host, "_user_for", lambda _: "fake")
    monkeypatch.setattr(sandbox_host, "browser_profile_dir", lambda _: udd)
    monkeypatch.setattr(sandbox_host, "chromium_executable", lambda _: "/fake/chromium")
    monkeypatch.setattr(streams, "sandbox_temp_dir", lambda _: str(tmp_path))
    monkeypatch.setattr(streams, "forward_port", lambda *a, **kw: 12345)
    monkeypatch.setattr(streams, "forward_failures", lambda *a: 0)
    monkeypatch.setattr(sandbox, "_live_devtools", lambda *a: next(found))
    monkeypatch.setattr(sandbox, "_start_chromium", lambda *a: (effects.append("launch") or 123, "1"))
    monkeypatch.setattr(sandbox, "_wait_for_port", lambda *a: (12345, "/devtools/browser/fake"))
    monkeypatch.setattr(streams, "run_in", lambda *a, **kw: effects.append("stop"))
    for attr in ("_active_sessions", "_session_last_activity", "_session_owner_homes"):
        monkeypatch.setattr(browser_session, attr, {})

    with lease.admitted_action():
        started = sandbox.create_sandbox_session("first")
    browser_session._active_sessions["first"] = started
    lease.acquire("viewer")
    browser_session_lifecycle.cleanup_single_session("first")
    assert browser_session._active_sessions["first"] == started
    assert effects == ["launch"]
    lease.release("viewer")
    browser_session_lifecycle.cleanup_single_session("first")
    with lease.admitted_action():
        attached = sandbox.create_sandbox_session("next-worker")
    assert attached["cdp_url"] == started["cdp_url"]
    assert effects == ["launch"]


def test_live_reconfiguration_cannot_launch_or_reap_in_the_exclusive_profile(monkeypatch, explicit_policy, capsys):
    from hermes_cli import cli_commands_mixin
    from tui_gateway import methods_browser

    def unexpected(*a, **kw):
        pytest.fail("live browser setup was reached")

    monkeypatch.setattr(cli_commands_mixin, "_normalize_cdp_url", unexpected)
    monkeypatch.setattr(methods_browser, "_err", lambda rid, code, message: (code, message), raising=False)
    monkeypatch.setattr(methods_browser, "_connect_local_default", unexpected)
    monkeypatch.setattr(methods_browser, "_reap_profile_browsers", unexpected)
    cli_commands_mixin._browser_connect(object(), "")
    cli_commands_mixin._browser_disconnect(object())
    assert "explicit release" in capsys.readouterr().out
    assert methods_browser._browser_connect(1, {})[0] == 5031
    assert methods_browser._browser_disconnect(1)[0] == 5031


@pytest.mark.platforms("posix")
def test_explicit_harness_attaches_separately_after_endpoint_replacement(monkeypatch, tmp_path, explicit_policy):
    from tools.browser_harness_dirs import daemon_identity

    identities = []

    def run_cli(cmd, code, env, timeout):
        identities.append(daemon_identity(env))
        return subprocess.CompletedProcess([], 0, "done", "")

    browser_use = _wire_browser_exec(monkeypatch, run_cli)
    env = {"BH_RUNTIME_DIR": str(tmp_path), "BH_TMP_DIR": str(tmp_path)}
    monkeypatch.setattr(browser_use, "_base_subprocess_env", lambda: dict(env))
    for port in (12345, 12345, 12346):
        env["BU_CDP_WS"] = f"ws://127.0.0.1:{port}/devtools/browser/fake"
        assert json.loads(browser_use.browser_exec("print(1)", task_id="relay"))["success"]
    assert identities[0] == identities[1]
    assert identities[0] != identities[2]
