"""Source template for the standalone Kaggle agent in submission.py."""

import base64
import platform
import struct
import sys
import tempfile
import time
import zlib
from functools import lru_cache
from pathlib import Path


_NATIVE_BLOBS = {}  # Filled by build_submission.py.
_OPENING_BYTES = b""  # Filled by build_submission.py.

_STANDARD = (6, 7, 4)
_TURN_LIMIT = 1.65
_MATE = 10_000
_INFINITY = _MATE + 1
_EXACT, _LOWER, _UPPER = range(3)


def _count(bits):
    return bin(bits).count("1")


_popcount = getattr(int, "bit_count", _count)


class _Geometry:
    def __init__(self, rows, columns, target):
        self.rows = rows
        self.columns = columns
        self.target = target
        self.stride = rows + 1
        self.size = rows * columns
        self.order = sorted(range(columns), key=lambda c: abs(2 * c - columns + 1))
        self.bottom = sum(1 << (c * self.stride) for c in range(columns))
        self.column_masks = tuple(((1 << rows) - 1) << (c * self.stride) for c in range(columns))
        self.board = self.bottom * ((1 << rows) - 1)
        self.first_parity = self.bottom * sum(1 << r for r in range(0, rows, 2))
        self.windows = () if target == 4 else self._windows()

    def _windows(self):
        windows = set()
        for column in range(self.columns):
            for row in range(self.rows):
                for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    end_column = column + dc * (self.target - 1)
                    end_row = row + dr * (self.target - 1)
                    if 0 <= end_column < self.columns and 0 <= end_row < self.rows:
                        windows.add(sum(
                            1 << ((column + dc * i) * self.stride + row + dr * i)
                            for i in range(self.target)
                        ))
        return tuple(windows)

    def encode(self, board, mark):
        current = occupied = 0
        for i, stone in enumerate(board):
            if stone:
                row, column = divmod(i, self.columns)
                bit = 1 << (column * self.stride + self.rows - row - 1)
                occupied |= bit
                if stone == mark:
                    current |= bit
        return current, occupied

    def landings(self, occupied):
        return (occupied + self.bottom) & self.board

    def wins(self, stones, occupied):
        if self.target == 4:
            gaps = (stones << 1) & (stones << 2) & (stones << 3)
            for step in (self.stride - 1, self.stride, self.stride + 1):
                pair = stones & (stones << step)
                gaps |= (pair << step) & ((stones << (3 * step)) | (stones >> step))
                gaps |= (pair >> (2 * step)) & ((stones >> (3 * step)) | (stones << step))
            return gaps & self.board & ~occupied

        enemy = stones ^ occupied
        gaps = 0
        for window in self.windows:
            if not window & enemy:
                empty = window & ~occupied
                if empty and not empty & (empty - 1):
                    gaps |= empty
        return gaps

    def reflect(self, bits):
        reflected = 0
        segment = (1 << self.stride) - 1
        for _ in range(self.columns):
            reflected = (reflected << self.stride) | (bits & segment)
            bits >>= self.stride
        return reflected

    def key(self, current, occupied):
        key = current + occupied
        mirrored = self.reflect(key)
        return (mirrored, True) if mirrored < key else (key, False)

    def first(self, moves):
        return next(c for c in self.order if self.column_masks[c] & moves)


@lru_cache(maxsize=8)
def _geometry(rows, columns, target):
    return _Geometry(rows, columns, target)


def _safe_moves(legal, enemy_wins):
    forced = legal & enemy_wins
    if forced & (forced - 1):
        return 0
    return (forced or legal) & ~(enemy_wins >> 1)


def _bind_native(path):
    import ctypes

    # The native tables are shared between calls. PyDLL keeps the GIL held.
    library = ctypes.PyDLL(str(path))
    library.cx_choose.argtypes = (
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_double,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64),
    )
    library.cx_choose.restype = ctypes.c_int
    library.cx_prove.argtypes = (
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_double, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64),
    )
    library.cx_prove.restype = ctypes.c_int
    library.cx_can_force.argtypes = (
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_double, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    )
    library.cx_can_force.restype = ctypes.c_int
    return library


@lru_cache(maxsize=1)
def _native_engine():
    machine = platform.machine().lower()
    blob = _NATIVE_BLOBS.get((sys.platform, "x86_64" if machine == "amd64" else machine))
    if blob is None:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="connectx-") as directory:
            path = Path(directory) / ("engine.dylib" if sys.platform == "darwin" else "engine.so")
            path.write_bytes(zlib.decompress(base64.b64decode(blob)))
            return _bind_native(path)
    except (ImportError, OSError, AttributeError, ValueError, zlib.error):
        return None


@lru_cache(maxsize=1)
def _opening_book():
    if not _OPENING_BYTES:
        return {}
    packed = zlib.decompress(base64.b64decode(_OPENING_BYTES))
    return dict(struct.iter_unpack("<QB", packed))


class _SearchTimeout(Exception):
    pass


class _PythonSearch:
    def __init__(self, geometry, deadline):
        self.geometry = geometry
        self.deadline = deadline
        self.nodes = 0
        self.table = {}

    def _evaluate(self, current, occupied, legal, enemy_wins, played):
        geometry = self.geometry
        other = current ^ occupied
        own_wins = geometry.wins(current, occupied)
        parity = geometry.first_parity
        if played & 1:
            parity ^= geometry.board
        center = geometry.column_masks[geometry.order[0]]
        return (26 * (_popcount(own_wins) - _popcount(enemy_wins))
                + 29 * (_popcount(own_wins & parity) - _popcount(enemy_wins & ~parity))
                + 32 * (_popcount(own_wins & legal) - _popcount(enemy_wins & legal))
                + 13 * (_popcount(own_wins & (legal << 1)) - _popcount(enemy_wins & (legal << 1)))
                + 7 * (_popcount(current & center) - _popcount(other & center)))

    def _search(self, current, occupied, played, depth, alpha, beta):
        self.nodes += 1
        if self.nodes % 64 == 0 and time.monotonic() >= self.deadline:
            raise _SearchTimeout

        geometry = self.geometry
        legal = geometry.landings(occupied)
        if not legal:
            return 0, -1
        other = current ^ occupied
        enemy_wins = geometry.wins(other, occupied)
        safe = _safe_moves(legal, enemy_wins)
        if not safe:
            return -_MATE + played + 2, geometry.first(legal)
        if played >= geometry.size - 2:
            return 0, geometry.first(safe)
        if depth == 0 and safe & (safe - 1):
            value = self._evaluate(current, occupied, legal, enemy_wins, played)
            return max(-900, min(900, value)), -1

        key = current + occupied
        entry = self.table.get(key)
        preferred = -1
        if entry is not None:
            old_depth, value, preferred, bound = entry
            if old_depth >= depth and (bound == _EXACT or
                    (bound == _LOWER and value >= beta) or (bound == _UPPER and value <= alpha)):
                return value, preferred

        moves = [column for column in geometry.order if safe & geometry.column_masks[column]]
        if preferred in moves:
            moves.remove(preferred)
            moves.insert(0, preferred)
        original_alpha = alpha
        best, best_column = -_INFINITY, moves[0]
        next_depth = max(0, depth - 1)
        for column in moves:
            move = safe & geometry.column_masks[column]
            score, _ = self._search(other, occupied | move, played + 1, next_depth, -beta, -alpha)
            score = -score
            if score > best:
                best, best_column = score, column
            alpha = max(alpha, best)
            if alpha >= beta:
                break

        if len(self.table) < 48_000 and (entry is None or entry[0] <= depth):
            bound = _UPPER if best <= original_alpha else _LOWER if best >= beta else _EXACT
            self.table[key] = depth, best, best_column, bound
        return best, best_column

    def choose(self, current, occupied, played, best):
        for depth in range(1, self.geometry.size - played + 1):
            if time.monotonic() >= self.deadline:
                break
            try:
                score, column = self._search(current, occupied, played, depth, -_INFINITY, _INFINITY)
            except _SearchTimeout:
                break
            best = column
            if score > 900:
                break
        return best


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def agent(observation, configuration):
    """Return a zero-based column for a Kaggle ConnectX turn."""
    start = time.monotonic()
    dimensions = (
        _field(configuration, "rows"),
        _field(configuration, "columns"),
        _field(configuration, "inarow"),
    )
    geometry = _geometry(*dimensions)
    current, occupied = geometry.encode(_field(observation, "board"), _field(observation, "mark"))
    legal = geometry.landings(occupied)
    if not occupied or geometry.target == 1 or geometry.target > max(dimensions[:2]) or not legal:
        return geometry.first(legal) if legal else 0

    wins = geometry.wins(current, occupied) & legal
    if wins:
        return geometry.first(wins)
    safe = _safe_moves(legal, geometry.wins(current ^ occupied, occupied))
    best = geometry.first(safe or legal)
    if not safe & (safe - 1):
        return best

    timeout = _field(configuration, "actTimeout", 2.0)
    if not isinstance(timeout, (int, float)):
        timeout = 2.0
    deadline = start + min(_TURN_LIMIT, 0.825 * max(0, timeout))
    played = _popcount(occupied)

    if dimensions == _STANDARD:
        key, reflected = geometry.key(current, occupied)
        column = _opening_book().get(key)
        if column is not None:
            if reflected:
                column = geometry.columns - 1 - column
            if safe & geometry.column_masks[column]:
                return column

        native = _native_engine()
        remaining_time = deadline - time.monotonic()
        if native is not None and remaining_time > 0:
            column = native.cx_choose(current, occupied, remaining_time, None, None)
            if 0 <= column < geometry.columns and safe & geometry.column_masks[column]:
                return column

    return _PythonSearch(geometry, deadline).choose(current, occupied, played, best)
