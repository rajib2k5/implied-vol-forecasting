"""Shared pytest configuration: no test may reach the network, and a test that tries fails.

The guard is installed in ``pytest_configure`` and removed in ``pytest_unconfigure``, so it covers
the whole session: collection (module-level code in test files), every fixture of every scope, and
every test, whether or not it asks for ``no_network`` (amended M1, review A). Review A found the
first version was a fixture that only the data tests requested, that patched only
``socket.socket.connect`` and ``socket.create_connection``, and that let ``connect_ex``, DNS
(``getaddrinfo``) and yfinance's HTTP client (``curl_cffi``, which is libcurl in C and never touches
Python's ``socket`` module) through.

Blocked routes, each tested against a listener on 127.0.0.1 in ``tests/test_network_guard.py``:

* ``socket.socket.connect`` and ``connect_ex`` for AF_INET and AF_INET6 (the class itself is
  patched, so instances created earlier, ``ssl.SSLSocket`` and everything built on them, such as
  ``requests``/``urllib3``, are covered);
* ``socket.socket.sendto`` and ``sendmsg`` for AF_INET and AF_INET6: UDP, as DNS resolvers use it,
  and TCP fast open (``MSG_FASTOPEN``), which connects inside the send. ``sendmsg`` was open until
  review B delivered a UDP datagram and made a TCP connection through it (amended M1, review B);
* ``socket.create_connection``;
* DNS: ``socket.getaddrinfo``, ``gethostbyname``, ``gethostbyname_ex``, ``gethostbyaddr`` and
  ``getnameinfo``. The last four were open until review B saw ``gethostbyname`` send a query to
  port 53 during a test (amended M1, review B);
* ``curl_cffi``: ``Curl.perform`` (every synchronous request) and ``AsyncCurl.add_handle`` (every
  asynchronous one), which is how yfinance 1.7 talks to Yahoo.

Every refusal is recorded, and the test in which it happened fails (amended M1, review B). Raising
``NetworkBlocked`` is not enough on its own: yfinance catches every exception under its default
``hide_exceptions``, logs "possibly delisted" and returns an empty frame, so a test that forgot to
give it a fake network would pass with no data. The first version of this docstring said no
library swallows the exception; review B showed yfinance does. Each refusal is recorded with what
was called, the node id and the phase, and charged as follows:

* during a test's setup, call or teardown: that phase's report is made a failure (setup and
  teardown show as errors), whatever the code under test did with the exception
  (``pytest_runtest_makereport``). A session or module fixture is charged to the test whose
  setup or teardown runs it;
* while a file is collected (module-level code): a collection error, which stops the session
  before any test runs (``pytest_make_collect_report``);
* anywhere else (another plugin's session hook, a thread still running between tests): listed
  under "network guard" in the terminal summary, and the exit status becomes 1
  (``pytest_sessionfinish``).

Only ``expect_blocked`` excuses a refusal. A test OF the guard makes the refused call inside
``with expect_blocked():``; the block also fails if nothing inside it was refused, so a route that
stops being blocked fails its test instead of passing quietly. A test that forgets to inject a
fake network therefore fails; it cannot pass by forgetting something.

``expect_blocked`` works only in the files named in ``EXPECT_BLOCKED_FILES``
(``test_network_guard.py`` and ``test_data_fetch.py``) and fails loudly anywhere else (M1 final
check). Before that, a test anywhere could wrap production code in ``expect_blocked()``, swallow
the refusal, and pass. Now such a use is recorded as a refusal of its own and charged to the test
like any other, so it fails even if the test catches the ``pytest.fail`` it raises.

Known limits, not closed:

* a **subprocess** runs without these patches (the pipeline and diagram tests start
  ``python -m ...`` and the diagram kit);
* ``_socket`` used directly is not seen. That includes ``socket.SocketType``, which is
  ``_socket.socket`` under a public name, and ``super(socket.socket, sock).connect``: a C type
  cannot be patched, so its methods connect (review B connected through ``SocketType``);
* a reference taken before the guard was installed (``from socket import create_connection`` in a
  module imported earlier) keeps the real function. ``ssl.create_connection`` is one; it is still
  refused, because the real ``create_connection`` resolves through the patched ``getaddrinfo``
  and connects through the patched ``connect`` (tested);
* C code other than libcurl that opens its own sockets (for example libzmq) is not seen;
* AF_UNIX sockets are allowed;
* code that runs before this file's ``pytest_configure`` or after its ``pytest_unconfigure`` is
  not covered, and a thread that outlives its test is charged to whatever is running when it
  tries (another test, or the session), so the failure may land on the wrong test.
"""

from __future__ import annotations

import contextlib
import socket
import threading
from dataclasses import dataclass
from pathlib import PurePosixPath

import pytest

# tests/test_network_guard.py runs inner pytest sessions against this file (amended M1, review B).
pytest_plugins = ("pytester",)


class NetworkBlocked(RuntimeError):
    """A test tried to reach the network.

    Not an OSError, so code that handles connection errors (retries, fallbacks) does not take it
    for one. It can still be swallowed: yfinance catches every exception under its default
    ``hide_exceptions`` (amended M1, review B), which is why every refusal is also recorded and
    fails the test without the exception's help.
    """


_INET = (socket.AF_INET, socket.AF_INET6)
_SOCKET_METHODS = ("connect", "connect_ex", "sendto", "sendmsg")
_SOCKET_FUNCTIONS = (
    "create_connection",
    "getaddrinfo",
    "gethostbyname",
    "gethostbyname_ex",
    "gethostbyaddr",
    "getnameinfo",
)


@dataclass(frozen=True)
class Attempt:
    """One refused call. ``nodeid`` is "" outside any test or collector."""

    call: str  # what was called, with its arguments, e.g. "socket.getaddrinfo('example.org', 443)"
    nodeid: str
    phase: str  # "setup", "call", "teardown", "collect" or "session"
    expected: bool  # made inside expect_blocked()

    def __str__(self) -> str:
        where = f"{self.phase} of {self.nodeid}" if self.nodeid else self.phase
        return f"{self.call}  [{where}]"


class NetworkGuard:
    """Replaces the network entry points with ones that record the call and raise ``NetworkBlocked``.

    ``remove`` puts them back. The record outlives ``remove``/``install``, so a test that lifts the
    guard for a moment does not lose what happened before.
    """

    def __init__(self) -> None:
        self._saved: list[tuple[object, str, object, bool]] = []
        self.active = False
        self.attempts: list[Attempt] = []  # every refusal of the session, in order
        self._pending: list[Attempt] = []  # unexpected refusals not yet charged to a report
        self.uncharged: list[Attempt] = []  # what pytest_sessionfinish found left over
        self._where = ("", "session")
        self._expecting = 0
        self._lock = threading.Lock()

    # -- recording ---------------------------------------------------------------------------

    def refuse(self, call: str) -> NetworkBlocked:
        """Record a refused call and return the exception to raise."""
        if len(call) > 300:
            call = call[:297] + "..."
        with self._lock:
            nodeid, phase = self._where
            attempt = Attempt(call, nodeid, phase, expected=self._expecting > 0)
            self.attempts.append(attempt)
            if not attempt.expected:
                self._pending.append(attempt)
        return NetworkBlocked(f"a test called {call}")

    @contextlib.contextmanager
    def during(self, nodeid: str, phase: str):
        """Charge refusals inside this block to (nodeid, phase). Nests: the outer one comes back."""
        with self._lock:
            previous, self._where = self._where, (nodeid, phase)
        try:
            yield
        finally:
            with self._lock:
                self._where = previous

    def take(self, nodeid: str | None = None, phase: str | None = None) -> list[Attempt]:
        """Remove and return the unexpected refusals charged to (nodeid, phase); all of them if None."""
        with self._lock:
            taken, kept = [], []
            for attempt in self._pending:
                mine = nodeid is None or (attempt.nodeid, attempt.phase) == (nodeid, phase)
                (taken if mine else kept).append(attempt)
            self._pending = kept
        return taken

    # -- patching ----------------------------------------------------------------------------

    def _patch(self, owner, name, replacement) -> None:
        # Remember whether the attribute was the owner's own or inherited (socket.socket.connect is
        # _socket.socket's), so remove() can put back exactly what was there.
        self._saved.append((owner, name, getattr(owner, name), name in vars(owner)))
        setattr(owner, name, replacement)

    def original(self, owner, name):
        """The function the guard replaced (for a test that lifts one layer to test another)."""
        for saved_owner, saved_name, function, _ in self._saved:
            if saved_owner is owner and saved_name == name:
                return function
        raise KeyError(f"the guard did not patch {owner!r}.{name}")

    def install(self) -> None:
        if self.active:
            return
        real = {name: getattr(socket.socket, name) for name in _SOCKET_METHODS if hasattr(socket.socket, name)}

        def refuse_inet(name):
            def method(sock, *args, **kwargs):
                if sock.family in _INET:
                    raise self.refuse(f"socket.socket.{name}{args!r}")
                return real[name](sock, *args, **kwargs)

            return method

        def refuse(what):
            def function(*args, **kwargs):
                raise self.refuse(f"{what}{args!r}")

            return function

        for name in real:
            self._patch(socket.socket, name, refuse_inet(name))
        for name in _SOCKET_FUNCTIONS:
            self._patch(socket, name, refuse(f"socket.{name}"))
        try:
            import curl_cffi.aio
            import curl_cffi.curl
        except ImportError:  # yfinance falls back to requests, which the socket patches cover
            pass
        else:
            self._patch(curl_cffi.curl.Curl, "perform", refuse("curl_cffi Curl.perform"))
            self._patch(curl_cffi.aio.AsyncCurl, "add_handle", refuse("curl_cffi AsyncCurl.add_handle"))
        self.active = True

    def remove(self) -> None:
        for owner, name, original, own in reversed(self._saved):
            if own:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self._saved.clear()
        self.active = False


GUARD = NetworkGuard()


#: The only test files (by file name, so the inner sessions of test_network_guard.py match too)
#: where ``expect_blocked`` may be used (M1 final check).
EXPECT_BLOCKED_FILES = ("test_network_guard.py", "test_data_fetch.py")


class expect_blocked:
    """``with expect_blocked() as blocked:`` marks the refusals inside as expected (amended M1, review B).

    For tests OF the guard only, in the files named in ``EXPECT_BLOCKED_FILES``; anywhere else it
    records the misuse as an unexpected refusal (so the test fails even if it catches what this
    raises) and fails the test (M1 final check). The refusals inside do not fail the test, and
    ``blocked.attempts`` lists them. The block fails if nothing inside it was refused, so a route
    that stops being blocked fails the test. It does not catch ``NetworkBlocked``: nest
    ``pytest.raises`` to require that the caller sees the exception.
    """

    def __enter__(self) -> expect_blocked:
        nodeid, phase = GUARD._where
        test_file = nodeid.split("::")[0]
        if PurePosixPath(test_file).name not in EXPECT_BLOCKED_FILES:
            misuse = Attempt(
                f"expect_blocked() in {test_file or 'no test'}: only {', '.join(EXPECT_BLOCKED_FILES)} may use it",
                nodeid,
                phase,
                expected=False,
            )
            with GUARD._lock:
                GUARD.attempts.append(misuse)
                GUARD._pending.append(misuse)
            pytest.fail(f"{misuse.call} (tests/conftest.py, M1 final check)")
        with GUARD._lock:
            GUARD._expecting += 1
            self._start = len(GUARD.attempts)
        self.attempts: list[Attempt] = []
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        with GUARD._lock:
            GUARD._expecting -= 1
            self.attempts = [a for a in GUARD.attempts[self._start :] if a.expected]
        if exc_type is None and not self.attempts:
            pytest.fail("expect_blocked(): nothing inside the block was refused; is this route still blocked?")
        return False


def _explain(attempts: list[Attempt]) -> str:
    lines = [f"network guard (tests/conftest.py) refused {len(attempts)} call(s):"]
    lines += [f"  {attempt}" for attempt in attempts]
    lines.append(
        "This fails even if the code under test caught NetworkBlocked and carried on (yfinance does). "
        "Give the code a fake network (tests/test_data_fetch.py replaces fetch.network_get); a test of "
        "the guard itself makes the call inside `with expect_blocked():`."
    )
    return "\n".join(lines)


def _charge(report, attempts: list[Attempt]) -> None:
    """Make ``report`` a failure that names the refused calls."""
    if not attempts:
        return
    text = _explain(attempts)
    if report.failed:
        report.sections.append(("network guard", text))
        return
    report.outcome = "failed"
    report.longrepr = text
    if hasattr(report, "wasxfail"):  # otherwise junitxml files the failure as a skipped xfail
        del report.wasxfail


def pytest_configure(config):
    GUARD.install()


def pytest_unconfigure(config):
    GUARD.remove()


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_setup(item):
    with GUARD.during(item.nodeid, "setup"):
        return (yield)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_call(item):
    with GUARD.during(item.nodeid, "call"):
        return (yield)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    with GUARD.during(item.nodeid, "teardown"):
        return (yield)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    # Outermost (tryfirst), so it sees the report after xfail handling and overrides it.
    report = yield
    _charge(report, GUARD.take(item.nodeid, call.when))
    return report


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_make_collect_report(collector):
    with GUARD.during(collector.nodeid, "collect"):
        report = yield
    _charge(report, GUARD.take(collector.nodeid, "collect"))
    return report


def pytest_sessionfinish(session, exitstatus):
    # Runs before the terminal summary is written, and main() returns session.exitstatus after it.
    GUARD.uncharged = GUARD.take()
    if GUARD.uncharged and session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter):
    if GUARD.uncharged:
        terminalreporter.section("network guard", sep="=", red=True, bold=True)
        terminalreporter.line(_explain(GUARD.uncharged))


@pytest.fixture
def no_network():
    """Kept for the tests that name it. The guard is on for the whole session; this only proves it."""
    assert GUARD.active, "the session-wide network guard is not installed"
    return GUARD
