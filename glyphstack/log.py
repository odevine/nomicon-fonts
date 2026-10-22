"""Progress reporting shared by the pipeline stages, live bars on a terminal and timestamped lines everywhere else

Stages report through `timed` and `Progress`, which send begin, step and end events to the active sink. Worker
processes started with `pool` send their log records and events to the parent over a queue, and the parent replays
them. The run log keeps every record at debug level, and the watchdog writes the stack of every thread in every
process to it when nothing has been reported for a while
"""
import atexit
import faulthandler
import itertools
import logging
import logging.handlers
import multiprocessing
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from rich.cells import cell_len
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.progress import Progress as Bars
from rich.table import Table
from rich.text import Text

log = logging.getLogger("glyphstack")
WATCHDOG_S = 300
LINE_FORMAT = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
FILE_FORMAT = logging.Formatter("%(asctime)s %(processName)s %(levelname)s %(message)s")


class _Lines:
    """Draws nothing, the timestamped log lines carry the progress"""

    def begin(self, key, label, total):
        pass

    def step(self, key, n, note):
        pass

    def end(self, key):
        pass


class _TwoLineBars(Bars):
    """Draws each task as its label and bar, with the count, times and note on an indented line below

    Labels share one width so the bars line up, and the bar takes the rest of the line, so a long note is cut at the
    edge of the terminal instead of squeezing the bar
    """

    spinner, bar, elapsed, remaining = SpinnerColumn(), BarColumn(bar_width=None), TimeElapsedColumn(), TimeRemainingColumn()

    def get_renderables(self):
        tasks = [t for t in self.tasks if t.visible]
        width = max((cell_len(t.description) for t in tasks), default=0)
        for t in tasks:
            top = Table.grid(padding=(0, 1), expand=True)
            top.add_column(no_wrap=True)
            top.add_column(width=width, no_wrap=True)
            top.add_column(ratio=1)
            top.add_row(self.spinner.render(t), t.description, self.bar.render(t))
            yield top
            indent = 2 + cell_len(t.description) - cell_len(t.description.lstrip()) + 2
            parts = [] if t.total is None else [MofNCompleteColumn().render(t), " "]
            parts += [self.elapsed.render(t), " elapsed"]
            if t.total is not None:
                parts += ["  ", self.remaining.render(t), " left"]
            if t.fields.get("note"):
                parts += ["  ", (t.fields["note"], "dim")]
            yield Text.assemble(" " * indent, *parts, no_wrap=True, overflow="ellipsis")


class _Live:
    """Draws a bar for each running Progress and a spinner for each running stage below the log lines"""

    def __init__(self, console):
        self.bars = _TwoLineBars(console=console, transient=True)
        self.tasks = {}
        self.bars.start()

    def begin(self, key, label, total):
        self.tasks[key] = self.bars.add_task(label, total=total, note="")

    def step(self, key, n, note):
        if key in self.tasks:
            self.bars.update(self.tasks[key], completed=n, note=note)

    def end(self, key):
        if key in self.tasks:
            self.bars.remove_task(self.tasks.pop(key))

    def stop(self):
        self.bars.stop()


class _Queue:
    """Sends events from a worker process to the parent, which draws them"""

    def __init__(self, q):
        self.q = q

    def begin(self, key, label, total):
        self.q.put(("begin", key, label, total))

    def step(self, key, n, note):
        self.q.put(("step", key, n, note))

    def end(self, key):
        self.q.put(("end", key))


class _State:
    sink = _Lines()
    live = False
    depth = 0
    last = time.monotonic()
    armed = 0.0
    timeout = 0
    dump = None
    logfile = None
    file_handler = None


_s = _State()
_ids = itertools.count()


def _pet():
    """Marks that something was reported and pushes the watchdog's deadline back, at most once a second"""
    now = time.monotonic()
    _s.last = now
    if _s.timeout and now - _s.armed >= 1:
        faulthandler.dump_traceback_later(_s.timeout, repeat=True, file=_s.dump)
        _s.armed = now


class _PetOnRecord(logging.Filter):
    def filter(self, record):
        if not getattr(record, "watchdog", False):
            _pet()
        return True


log.addFilter(_PetOnRecord())


def _watch():
    """Warns on the console once for every stretch with nothing reported for longer than the watchdog timeout"""
    warned = None
    where = _s.logfile or "stderr"
    while True:
        time.sleep(min(5.0, _s.timeout / 4))
        idle = time.monotonic() - _s.last
        if idle >= _s.timeout and warned != _s.last:
            warned = _s.last
            log.warning("nothing reported for %.0fs, the stack of every thread is written to %s", idle, where, extra={"watchdog": True})


class _Console(logging.Filter):
    """Passes records at or above level, and records marked for the console by `announce`"""

    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record):
        return record.levelno >= self.level or getattr(record, "console", False)


def announce(msg, *args):
    """Logs a line that is shown on the console even when the console shows only warnings"""
    log.info(msg, *args, extra={"console": True})


def _crash(hook):
    def excepthook(kind, value, tb):
        if _s.file_handler:
            _s.file_handler.handle(log.makeRecord(log.name, logging.CRITICAL, __file__, 0, "stopped by an error", (), (kind, value, tb)))
        hook(kind, value, tb)
    return excepthook


def setup(verbose=False, logfile=None, watchdog=0):
    """Starts reporting for a command line run, call once from its entry point

    On a terminal stages draw live bars and the console shows only warnings, errors and announced lines. Elsewhere
    stages log an info line every few seconds. Verbose shows every record on the console. The logfile gets every record
    at debug level, and with watchdog seconds the stack of every thread goes to it when nothing is reported for that long
    """
    console = Console(stderr=True)
    _s.live = console.is_terminal
    if _s.live:
        _s.sink = _Live(console)
        atexit.register(_s.sink.stop)
        shown = RichHandler(console=console, show_path=False, log_time_format="%H:%M:%S")
    else:
        _s.sink = _Lines()
        shown = logging.StreamHandler(sys.stderr)
        shown.setFormatter(LINE_FORMAT)
    shown.addFilter(_Console(logging.DEBUG if verbose else logging.WARNING if _s.live else logging.INFO))
    handlers = [shown]
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        _s.logfile = str(logfile)
        _s.file_handler = logging.FileHandler(logfile)
        _s.file_handler.setFormatter(FILE_FORMAT)
        handlers.append(_s.file_handler)
        sys.excepthook = _crash(sys.excepthook)
    log.handlers[:] = handlers
    log.propagate = False
    log.setLevel(logging.DEBUG)
    if watchdog:
        _s.timeout = watchdog
        _s.dump = open(logfile, "a") if logfile else sys.stderr
        atexit.register(faulthandler.cancel_dump_traceback_later)
        threading.Thread(target=_watch, daemon=True, name="watchdog").start()
        _pet()


def add_args(ap):
    """Adds the console, run log and watchdog options read by `start`"""
    ap.add_argument("-v", "--verbose", action="store_true", help="show every message on the console, not only warnings and errors")
    ap.add_argument("--log", help="run log with every message at debug level, default a timestamped file in a logs folder of the output")
    ap.add_argument("--watchdog", type=float, default=WATCHDOG_S, help="seconds with nothing reported before the stack of every thread is written to the run log, 0 for never")


def start(a, folder, name):
    """Calls setup with the options from `add_args`, the run log defaulting to folder/logs/name-time.log, and returns its path"""
    path = Path(a.log) if a.log else Path(folder) / "logs" / f"{name}-{datetime.now():%Y%m%d-%H%M%S}.log"
    setup(a.verbose, path, a.watchdog)
    announce("run log %s", path)
    return path


def _begin(label, total):
    key = f"{os.getpid()}.{next(_ids)}"
    _s.sink.begin(key, "  " * _s.depth + label, total)
    _s.depth += 1
    _pet()
    return key


def _end(key):
    _s.depth -= 1
    _s.sink.end(key)


@contextmanager
def timed(label):
    """Shows a spinner while a stage runs and logs how long it took"""
    log.log(logging.DEBUG if _s.live else logging.INFO, "%s ...", label)
    key = _begin(label, None)
    t0 = time.time()
    try:
        yield
    finally:
        _end(key)
    log.info("%s done in %.1fs", label, time.time() - t0)


class Progress:
    """Counts finished items of a known total, drawn as a bar on a terminal and logged at most every few seconds elsewhere"""

    def __init__(self, label, total, every=10.0):
        self.label, self.total, self.every = label, total, every
        self.n, self.t0, self.last = 0, time.time(), time.time()
        self.key = _begin(label, total) if total else None

    def step(self, note=""):
        self.n += 1
        now = time.time()
        if self.key:
            _s.sink.step(self.key, self.n, note)
        done = self.n == self.total
        loud = done or (not _s.live and now - self.last >= self.every)
        if loud:
            self.last = now
        left = (now - self.t0) / self.n * (self.total - self.n)
        log.log(logging.INFO if loud else logging.DEBUG, "%s %d/%d%s, %.0fs elapsed, about %.0fs left",
                self.label, self.n, self.total, f" ({note})" if note else "", now - self.t0, left)
        if done and self.key:
            _end(self.key)
            self.key = None


def _listen(q):
    """Replays worker records and events in this process until the pool sends None"""
    while (item := q.get()) is not None:
        if isinstance(item, logging.LogRecord):
            log.handle(item)
        else:
            getattr(_s.sink, item[0])(*item[1:])
            _pet()


def _worker_init(q, live, depth, timeout, logfile, initializer, initargs):
    log.handlers[:] = [logging.handlers.QueueHandler(q)]
    log.propagate = False
    log.setLevel(logging.DEBUG)
    _s.sink, _s.live, _s.depth = _Queue(q), live, depth
    if timeout:
        _s.timeout, _s.dump = timeout, open(logfile, "a") if logfile else sys.stderr
    if initializer:
        initializer(*initargs)


def _watched(fn, *args, **kwargs):
    """Runs one task in a worker with its watchdog armed, so an idle worker waiting for work never dumps"""
    _pet()
    try:
        return fn(*args, **kwargs)
    finally:
        if _s.timeout:
            faulthandler.cancel_dump_traceback_later()
            _s.armed = 0.0


class _Pool:
    def __init__(self, ex):
        self.ex = ex

    def submit(self, fn, *args, **kwargs):
        return self.ex.submit(_watched, fn, *args, **kwargs)


@contextmanager
def pool(workers, initializer=None, initargs=()):
    """A process pool whose workers log and draw progress through this process, with the watchdog running in each"""
    q = multiprocessing.get_context().Queue()
    listener = threading.Thread(target=_listen, args=(q,), daemon=True, name="worker-log")
    listener.start()
    try:
        cfg = (q, _s.live, _s.depth, _s.timeout, _s.logfile, initializer, initargs)
        with ProcessPoolExecutor(workers, initializer=_worker_init, initargs=cfg) as ex:
            yield _Pool(ex)
    finally:
        q.put(None)
        listener.join()
