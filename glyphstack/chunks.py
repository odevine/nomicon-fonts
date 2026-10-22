"""Many same-shaped images kept on disk in compressed chunks, so a glyph with thousands of instances is never held in memory whole"""
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

CHUNK_BYTES = 64 << 20


class Writer:
    """Appends images to a store folder, replacing what was there, and compresses full chunks on a background thread

    meta is written to the index with the count and shape, for example the key of each image
    """

    def __init__(self, folder, dtype=np.float16, meta=None):
        self.folder, self.dtype, self.meta = Path(folder), np.dtype(dtype), meta or {}
        if self.folder.exists():
            shutil.rmtree(self.folder)
        self.folder.mkdir(parents=True)
        self.buf, self.n, self.shape, self.chunk = [], 0, None, None
        self.pool, self.pending = ThreadPoolExecutor(1), []

    def add(self, img):
        if self.shape is None:
            self.shape = tuple(img.shape)
            self.chunk = max(1, CHUNK_BYTES // (int(np.prod(self.shape)) * self.dtype.itemsize))
        self.buf.append(np.asarray(img, self.dtype))
        self.n += 1
        if len(self.buf) == self.chunk:
            self._flush()

    def _flush(self):
        if self.buf:
            path, a = self.folder / f"{(self.n - 1) // self.chunk:05d}.npz", np.stack(self.buf)
            # at most one chunk compressing while the next one fills
            for f in self.pending:
                f.result()
            self.pending = [self.pool.submit(np.savez_compressed, path, a=a)]
            self.buf = []

    def close(self):
        self._flush()
        for f in self.pending:
            f.result()
        self.pool.shutdown()
        (self.folder / "index.json").write_text(json.dumps(dict(n=self.n, chunk=self.chunk, shape=self.shape, dtype=self.dtype.name, **self.meta)))
        return Store(self.folder)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.close()
        else:
            self.pool.shutdown()


class Store:
    """Read access to a folder written by Writer, images come back as float32

    Indexing keeps the last chunk read so a run of nearby images reads it once, take() keeps nothing
    """

    def __init__(self, folder):
        self.folder = Path(folder)
        self.index = json.loads((self.folder / "index.json").read_text())
        self.n, self.chunk = self.index["n"], self.index["chunk"]
        self.shape = tuple(self.index["shape"]) if self.index["shape"] else None
        self._cached = (None, None)

    def __len__(self):
        return self.n

    def _read(self, c):
        return np.load(self.folder / f"{c:05d}.npz")["a"]

    def __getitem__(self, i):
        c = i // self.chunk
        if self._cached[0] != c:
            self._cached = (c, self._read(c))
        return self._cached[1][i % self.chunk].astype(np.float32)

    def __iter__(self):
        for i in range(self.n):
            yield self[i]

    def take(self, idx):
        """The images at the given indices as one float32 array, in the order given, reading each chunk once"""
        idx = np.asarray(idx, int)
        out = np.empty((len(idx), *self.shape), np.float32)
        for c in np.unique(idx // self.chunk):
            sel = np.flatnonzero(idx // self.chunk == c)
            out[sel] = self._read(int(c))[idx[sel] % self.chunk]
        return out
