import logging
import os
import subprocess
import sys
from concurrent.futures import as_completed
from pathlib import Path

from glyphstack import log as glog
from glyphstack.log import Progress, log

ROOT = Path(__file__).parent.parent


class Recorder:
    def __init__(self):
        self.events = []

    def begin(self, key, label, total):
        self.events.append(("begin", label, total))

    def step(self, key, n, note):
        self.events.append(("step", n, note))

    def end(self, key):
        self.events.append(("end",))


def work(n):
    prog = Progress("items", n)
    for i in range(n):
        prog.step(f"item {i}")
    log.info("worker counted %d", n)
    return n


def test_pool_replays_worker_records_and_progress(monkeypatch, caplog):
    rec = Recorder()
    monkeypatch.setattr(glog._s, "sink", rec)
    with caplog.at_level(logging.INFO, logger="glyphstack"), glog.pool(2) as ex:
        got = sorted(f.result() for f in as_completed([ex.submit(work, 3), ex.submit(work, 2)]))
    assert got == [2, 3]
    assert "worker counted 3" in caplog.text and "items 3/3" in caplog.text
    kinds = [e[0] for e in rec.events]
    assert kinds.count("begin") == 2 and kinds.count("end") == 2 and kinds.count("step") == 5


def run(code, logfile, **env):
    env = {k: v for k, v in os.environ.items() if k not in ("FORCE_COLOR", "TTY_COMPATIBLE")} | env
    return subprocess.run([sys.executable, "-c", code, str(logfile)], cwd=ROOT, capture_output=True, text=True, timeout=60, env=env)


LEVELS = "import sys\nfrom glyphstack.log import announce, log, setup\nsetup(False, sys.argv[1])\nlog.info('quiet')\nlog.warning('loud')\nannounce('shown')\n"


def test_terminal_console_shows_warnings_and_announced_lines(tmp_path):
    r = run(LEVELS, tmp_path / "run.log", FORCE_COLOR="1")
    assert "loud" in r.stderr and "shown" in r.stderr and "quiet" not in r.stderr
    assert all(w in (tmp_path / "run.log").read_text() for w in ("quiet", "loud", "shown"))


def test_piped_console_keeps_info_lines(tmp_path):
    r = run(LEVELS, tmp_path / "run.log")
    assert all(w in r.stderr for w in ("quiet", "loud", "shown"))


def test_watchdog_warns_and_writes_stacks(tmp_path):
    logfile = tmp_path / "run.log"
    r = run("import sys, time\nfrom glyphstack.log import setup, log\nsetup(False, sys.argv[1], 1)\nlog.info('started')\ntime.sleep(3)\n", logfile)
    text = logfile.read_text()
    assert "nothing reported for" in r.stderr
    assert "started" in text and "most recent call first" in text


def test_crash_traceback_lands_in_run_log(tmp_path):
    logfile = tmp_path / "run.log"
    r = run("import sys\nfrom glyphstack.log import setup\nsetup(False, sys.argv[1])\nraise ValueError('bad scan')\n", logfile)
    assert r.returncode != 0
    text = logfile.read_text()
    assert "stopped by an error" in text and "ValueError: bad scan" in text
