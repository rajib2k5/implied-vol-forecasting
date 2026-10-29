"""The session-wide network guard (tests/conftest.py) blocks every route it claims to, and a refusal fails its test.

Amended M1, review A, then M1, review B.

Each route test opens a real listener on 127.0.0.1 (bind and listen are allowed: they reach
nothing) and then tries one route to it. A route that is not blocked would connect, so each test
fails on a guard that lost that patch: review A's mutant V1 (no ``socket.socket.connect`` patch)
passed all 578 tests of the first version, whose only guard test exercised
``socket.create_connection``. Review B then delivered a UDP datagram and a TCP fast-open connection
through ``sendmsg``, and resolved names through ``gethostbyname`` (a query to port 53); those
routes are now patched and tested here too.

Every refusal below is made inside ``expect_blocked()``, which excuses it and fails if nothing was
refused. Outside that block a refusal fails the test even when the exception is swallowed; the
last tests here prove that by running inner pytest sessions against a copy of the conftest.

None of these tests asks for the ``no_network`` fixture: the guard must be on without it.
"""

from __future__ import annotations

import re
import socket
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import conftest
from conftest import GUARD, NetworkBlocked, expect_blocked


@pytest.fixture
def listener():
    """A TCP server on 127.0.0.1 that accepts connections in a thread; yields its (host, port)."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    server.settimeout(0.2)
    accepted = []
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except (TimeoutError, OSError):
                continue
            accepted.append(conn)
            try:
                conn.sendall(b"HTTP/1.0 200 OK\r\nContent-Length: 2\r\n\r\nok")
            except OSError:
                pass
            conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield server.getsockname(), accepted
    stop.set()
    thread.join(timeout=2)
    server.close()


@pytest.fixture
def udp_listener():
    """A UDP socket bound to 127.0.0.1; a datagram that got through can be read from it."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
        server.bind(("127.0.0.1", 0))
        server.settimeout(0.3)
        yield server


def nothing_arrived(server) -> bool:
    try:
        server.recv(64)
    except TimeoutError:
        return True
    return False


def test_the_guard_is_on_for_a_test_that_did_not_ask_for_it():
    assert GUARD.active


def test_socket_connect_is_blocked(listener):
    address, accepted = listener
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock, expect_blocked(), pytest.raises(NetworkBlocked):
        sock.connect(address)
    assert accepted == []


def test_socket_connect_ex_is_blocked(listener):
    address, accepted = listener
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock, expect_blocked(), pytest.raises(NetworkBlocked):
        sock.connect_ex(address)
    assert accepted == []


def test_udp_sendto_is_blocked(udp_listener):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, expect_blocked(), pytest.raises(NetworkBlocked):
        sock.sendto(b"x", udp_listener.getsockname())
    assert nothing_arrived(udp_listener)


def test_udp_sendmsg_is_blocked(udp_listener):
    # Review B delivered this datagram through the first guard (amended M1, review B).
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, expect_blocked(), pytest.raises(NetworkBlocked):
        sock.sendmsg([b"x"], [], 0, udp_listener.getsockname())
    assert nothing_arrived(udp_listener)


@pytest.mark.skipif(not hasattr(socket, "MSG_FASTOPEN"), reason="no TCP fast open on this platform")
def test_tcp_fast_open_through_sendmsg_is_blocked(listener):
    # MSG_FASTOPEN connects inside the send; review B's listener accepted it (amended M1, review B).
    address, accepted = listener
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock, expect_blocked(), pytest.raises(NetworkBlocked):
        sock.sendmsg([b"GET / HTTP/1.0\r\n\r\n"], [], socket.MSG_FASTOPEN, address)
    time.sleep(0.3)  # the accept loop polls every 0.2 s
    assert accepted == []


def test_create_connection_is_blocked(listener):
    address, accepted = listener
    with expect_blocked(), pytest.raises(NetworkBlocked):
        socket.create_connection(address, timeout=1)
    assert accepted == []


def test_create_connection_is_blocked_by_its_own_patch(listener, monkeypatch):
    # create_connection resolves through socket.getaddrinfo and connects through
    # socket.socket.connect, which are refused on their own; so the test above passes without
    # create_connection's patch (review B: all 61 tests did). Lifting those two layers here makes
    # this test fail if that patch goes (amended M1, review B). Numeric address: no DNS.
    address, accepted = listener
    monkeypatch.setattr(socket, "getaddrinfo", GUARD.original(socket, "getaddrinfo"))
    monkeypatch.setattr(socket.socket, "connect", GUARD.original(socket.socket, "connect"))
    with expect_blocked() as blocked, pytest.raises(NetworkBlocked):
        socket.create_connection(address, timeout=1)
    time.sleep(0.3)
    assert accepted == []
    assert [a.call.split("(")[0] for a in blocked.attempts] == ["socket.create_connection"]


def test_a_reference_taken_before_the_guard_is_still_refused(listener, monkeypatch):
    # ssl does `from socket import create_connection`, so if ssl was imported before the guard it
    # keeps the real function (review B found ssl.create_connection was). Make that so here, and
    # show the real function is still refused: it resolves through the patched getaddrinfo.
    import ssl

    address, accepted = listener
    monkeypatch.setattr(ssl, "create_connection", GUARD.original(socket, "create_connection"))
    with expect_blocked(), pytest.raises(NetworkBlocked):
        ssl.get_server_certificate(address, timeout=1)
    assert accepted == []


def test_dns_lookup_is_blocked():
    with expect_blocked(), pytest.raises(NetworkBlocked):
        socket.getaddrinfo("127.0.0.1", 80)
    with expect_blocked(), pytest.raises(NetworkBlocked):
        socket.getaddrinfo("cdn.cboe.com", 443)


@pytest.mark.parametrize(
    ("name", "args"),
    [
        # Numeric arguments, so that if the patch is missing the call answers locally and sends no
        # DNS query; the test then fails because nothing was refused.
        ("gethostbyname", ("127.0.0.1",)),
        ("gethostbyname_ex", ("127.0.0.1",)),
        ("gethostbyaddr", ("127.0.0.1",)),
        ("getnameinfo", (("127.0.0.1", 80), socket.NI_NUMERICHOST | socket.NI_NUMERICSERV)),
    ],
)
def test_the_older_dns_functions_are_blocked(name, args):
    # Review B saw gethostbyname send a query to port 53 through the first guard (amended M1, review B).
    with expect_blocked() as blocked, pytest.raises(NetworkBlocked):
        getattr(socket, name)(*args)
    assert [a.call.split("(")[0] for a in blocked.attempts] == [f"socket.{name}"]


def test_requests_the_cboe_client_is_blocked(listener):
    import requests

    (host, port), accepted = listener
    with expect_blocked(), pytest.raises(NetworkBlocked):
        requests.get(f"http://{host}:{port}/", timeout=2)
    assert accepted == []


def test_curl_cffi_the_yfinance_client_is_blocked(listener):
    # yfinance 1.7 fetches through curl_cffi (yfinance/_http.py), i.e. libcurl in C: no Python socket.
    curl_requests = pytest.importorskip("curl_cffi.requests")
    (host, port), accepted = listener
    with expect_blocked(), pytest.raises(NetworkBlocked):
        curl_requests.get(f"http://{host}:{port}/", timeout=2)
    assert accepted == []


def test_curl_cffi_async_requests_are_blocked(listener):
    import asyncio

    curl_requests = pytest.importorskip("curl_cffi.requests")
    (host, port), accepted = listener

    async def fetch():
        async with curl_requests.AsyncSession() as session:
            return await session.get(f"http://{host}:{port}/", timeout=2)

    with expect_blocked(), pytest.raises(NetworkBlocked):
        asyncio.run(fetch())
    assert accepted == []


def test_unix_sockets_still_work(tmp_path):
    # AF_UNIX reaches no network; asyncio and some libraries use socketpair() internally, and
    # multiprocessing passes file descriptors with sendmsg on AF_UNIX.
    a, b = socket.socketpair()
    with a, b:
        a.sendall(b"hi")
        assert b.recv(2) == b"hi"
        a.sendmsg([b"yo"])
        assert b.recv(2) == b"yo"


def test_removing_the_guard_restores_the_real_functions(listener, udp_listener):
    # remove() must put back exactly what was there, or the rest of the session would run unguarded
    # or broken. Reinstalled at once, inside the test. Nothing here leaves the machine.
    address, accepted = listener
    GUARD.remove()
    try:
        with socket.create_connection(address, timeout=2) as sock:
            assert sock.recv(64).startswith(b"HTTP/1.0 200")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendmsg([b"x"], [], 0, udp_listener.getsockname())
        assert udp_listener.recv(8) == b"x"
        assert socket.gethostbyname("127.0.0.1") == "127.0.0.1"
        assert "connect" not in vars(socket.socket) and "sendmsg" not in vars(socket.socket)  # inherited again
    finally:
        GUARD.install()
    assert GUARD.active and "connect" in vars(socket.socket)
    with expect_blocked(), pytest.raises(NetworkBlocked):
        socket.gethostbyname("127.0.0.1")


# --- a refusal fails its test even when the exception is swallowed (amended M1, review B) ---------
#
# These run pytest in a subprocess, on a temporary directory holding a copy of tests/conftest.py and
# an inner test file. A subprocess, not pytester's in-process run: the inner session must install
# its own guard on a fresh `socket` module and decide its own exit status, which an in-process run
# would share with this session's guard (already installed on the same module).
#
# The refused calls cannot leave the machine even if the guard were broken (as in a mutation run):
# getaddrinfo is given a numeric address, and yfinance a proxy on 127.0.0.1 that refuses.


def run_inner(pytester, *args, **files):
    pytester.makeini("[pytest]\n")  # the inner rootdir is the temporary directory
    pytester.makeconftest(Path(conftest.__file__).read_text(encoding="utf-8"))
    pytester.makepyfile(**files)
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", "-rA", *args, timeout=120)


def junit_outcomes(path: Path) -> dict[str, set[str]]:
    """Each test case's name -> the tags under it ("failure", "error", "skipped"; empty = passed)."""
    root = ET.parse(path).getroot()
    return {case.get("name"): {child.tag for child in case} - {"system-out", "system-err"} for case in root.iter("testcase")}


SWALLOWING = '''
import socket

import pytest

from conftest import NetworkBlocked, expect_blocked


def touch():
    try:
        socket.getaddrinfo("127.0.0.1", 9)
    except NetworkBlocked:
        pass  # what yfinance does with it


@pytest.fixture
def touches_in_setup():
    touch()
    yield


@pytest.fixture
def touches_in_teardown():
    yield
    touch()


def test_swallowed_in_the_test():
    touch()


def test_swallowed_in_setup(touches_in_setup):
    pass


def test_swallowed_in_teardown(touches_in_teardown):
    pass


@pytest.mark.xfail(reason="xfail does not excuse it")
def test_swallowed_in_an_xfail_test():
    touch()


def test_swallowed_then_skipped():
    touch()
    pytest.skip("skipping does not excuse it")


def test_swallowed_then_failed():
    touch()
    assert False, "its own failure"


def test_clean():
    pass


def test_an_expected_refusal_passes():
    with expect_blocked(), pytest.raises(NetworkBlocked):
        socket.getaddrinfo("127.0.0.1", 9)


def test_expect_blocked_with_nothing_refused_fails():
    with expect_blocked():
        socket.socketpair()[0].close()
'''


def test_a_swallowed_refusal_fails_the_test_in_every_phase(pytester):
    # Named like this file: expect_blocked works only in the files EXPECT_BLOCKED_FILES names.
    result = run_inner(pytester, "--junitxml=inner.xml", test_network_guard=SWALLOWING)
    result.assert_outcomes(passed=3, failed=5, errors=2)
    assert junit_outcomes(pytester.path / "inner.xml") == {
        "test_swallowed_in_the_test": {"failure"},
        "test_swallowed_in_setup": {"error"},
        "test_swallowed_in_teardown": {"error"},
        "test_swallowed_in_an_xfail_test": {"failure"},
        "test_swallowed_then_skipped": {"failure"},
        "test_swallowed_then_failed": {"failure"},
        "test_clean": set(),
        "test_an_expected_refusal_passes": set(),
        "test_expect_blocked_with_nothing_refused_fails": {"failure"},
    }
    out = result.stdout.str()
    for phase, test in [
        ("call", "test_swallowed_in_the_test"),
        ("setup", "test_swallowed_in_setup"),
        ("teardown", "test_swallowed_in_teardown"),
        ("call", "test_swallowed_in_an_xfail_test"),
        ("call", "test_swallowed_then_skipped"),
        ("call", "test_swallowed_then_failed"),
    ]:
        assert f"socket.getaddrinfo('127.0.0.1', 9)  [{phase} of test_network_guard.py::{test}]" in out
    assert "nothing inside the block was refused" in out
    # A test that also failed on its own keeps its own report; the guard's is added beside it.
    assert "its own failure" in out and "- network guard -" in out


MISUSING = '''
import socket

import pytest

from conftest import NetworkBlocked, expect_blocked


def production_code():
    try:
        socket.getaddrinfo("127.0.0.1", 9)
    except NetworkBlocked:
        pass  # what yfinance does with it


def test_wraps_production_code_in_expect_blocked():
    with expect_blocked():
        production_code()


def test_also_swallows_what_expect_blocked_raises():
    try:
        with expect_blocked():
            production_code()
    except BaseException:
        pass


def test_wraps_code_that_never_touches_the_network():
    try:
        with expect_blocked():
            pass
    except BaseException:
        pass


def test_clean():
    pass
'''


def test_expect_blocked_outside_the_guard_and_fetch_tests_fails_loudly(pytester):
    # M1 final check: a test that wrapped production code in expect_blocked() and swallowed the
    # refusal passed. Outside the files EXPECT_BLOCKED_FILES names it now fails, even when the test
    # also catches what expect_blocked raises.
    result = run_inner(pytester, "--junitxml=inner.xml", test_elsewhere=MISUSING)
    result.assert_outcomes(passed=1, failed=3)
    assert junit_outcomes(pytester.path / "inner.xml") == {
        "test_wraps_production_code_in_expect_blocked": {"failure"},
        "test_also_swallows_what_expect_blocked_raises": {"failure"},
        "test_wraps_code_that_never_touches_the_network": {"failure"},
        "test_clean": set(),
    }
    assert "expect_blocked() in test_elsewhere.py: only test_network_guard.py, test_data_fetch.py may use it" in result.stdout.str()


def test_a_swallowed_refusal_while_collecting_is_a_collection_error(pytester):
    result = run_inner(
        pytester,
        test_collected='''
import socket

from conftest import NetworkBlocked

try:
    socket.getaddrinfo("127.0.0.1", 9)
except NetworkBlocked:
    pass


def test_never_runs():
    pass
''',
    )
    result.assert_outcomes(errors=1)
    assert result.ret == pytest.ExitCode.INTERRUPTED
    assert "socket.getaddrinfo('127.0.0.1', 9)  [collect of test_collected.py]" in result.stdout.str()


def test_a_swallowed_refusal_outside_any_test_fails_the_session(pytester):
    pytester.makepyfile(
        outside_plugin='''
import socket


def touch():
    try:
        socket.getaddrinfo("127.0.0.1", 9)
    except Exception:
        pass


def pytest_sessionstart(session):
    touch()


def pytest_collection_finish(session):
    touch()  # after collection: charged to the session, not to the last file collected
'''
    )
    result = run_inner(pytester, "-p", "outside_plugin", test_fine="def test_fine():\n    pass\n")
    result.assert_outcomes(passed=1)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(
        [
            "*= network guard =*",
            "*refused 2 call(s):",
            "  socket.getaddrinfo('127.0.0.1', 9)  [[]session[]]",
            "  socket.getaddrinfo('127.0.0.1', 9)  [[]session[]]",
        ]
    )


def test_yfinance_swallows_the_refusal_and_the_test_still_fails(pytester):
    # Review B's N4: yfinance catches NetworkBlocked under its default hide_exceptions and returns
    # an empty frame, so a test that forgot the fake network passed. It does so once the ticker's
    # time zone is cached, as it is after any real fetch (review B's run had it cached); on a cold
    # cache it asks `info` for the zone, which raises, at most twice per process. The inner test
    # warms its own cache (pytester's HOME is a temporary directory). Its own assertion holds (the
    # frame is empty), so it fails only because of the guard's record. The guard refuses before
    # any connection, so this needs no network and does not wait on retries.
    pytest.importorskip("yfinance")
    started = time.monotonic()
    result = run_inner(
        pytester,
        test_forgot='''
def test_forgot_the_fake_network(tmp_path):
    import socket

    import yfinance
    from yfinance import cache

    yfinance.set_tz_cache_location(str(tmp_path))
    cache.get_tz_cache().store("^VIX", "America/New_York")  # as after an earlier real fetch
    with socket.socket() as dead:  # bound, not listening: a proxy that refuses every connection
        dead.bind(("127.0.0.1", 0))
        yfinance.config.network.proxy = "http://127.0.0.1:%d" % dead.getsockname()[1]
        frame = yfinance.Ticker("^VIX").history(period="5d")
    assert frame.empty  # swallowed: no exception, no data
''',
    )
    elapsed = time.monotonic() - started
    result.assert_outcomes(failed=1)
    out = result.stdout.str()
    assert "curl_cffi Curl.perform" in out and "[call of test_forgot.py::test_forgot_the_fake_network]" in out
    assert "possibly delisted" in out  # yfinance's log line for the swallowed refusal
    # No exception reached the test: the failure is the guard's report, with no "E   ..." lines.
    assert not re.search(r"^E\s", out, re.MULTILINE), out
    assert elapsed < 60, f"the inner yfinance run took {elapsed:.1f} s"
