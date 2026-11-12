"""Generate an opening book by solving positions reached by either seat."""

import argparse
import ctypes
import json
import math
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from agent import _bind_native, _geometry, _safe_moves
from build_submission import ROOT, build_host


GEOMETRY = _geometry(6, 7, 4)
LIBRARY = None
UNKNOWN = 2


def replay(history):
    current = occupied = 0
    for character in history:
        column = int(character)
        move = GEOMETRY.landings(occupied) & GEOMETRY.column_masks[column]
        if not move or move & GEOMETRY.wins(current, occupied):
            raise ValueError(f"Opening history ends or leaves a legal game: {history}")
        current ^= occupied
        occupied |= move
    return current, occupied


def canonical(current, occupied, history):
    key, reflected = GEOMETRY.key(current, occupied)
    if reflected:
        current, occupied = GEOMETRY.reflect(current), GEOMETRY.reflect(occupied)
        history = "".join(str(6 - int(c)) for c in history)
    return key, (current, occupied, history), reflected


def read_opening_rows(paths):
    entries = {}
    for path in paths:
        for row in json.loads(path.read_text(encoding="utf-8")):
            history = row["history"]
            if type(row["move"]) is not int or row["move"] not in range(7) or row.get("outcome") not in (None, -1, 0, 1):
                raise ValueError(f"Invalid opening seed: {row}")
            key, (_, occupied, history), reflected = canonical(*replay(history), history)
            move = 6 - row["move"] if reflected else row["move"]
            if not (GEOMETRY.landings(occupied) & GEOMETRY.column_masks[move]):
                raise ValueError(f"Seed moves in a full column: {row}")
            result = row.get("outcome")
            previous = entries.get(key)
            if previous is None or (result is not None and (previous["outcome"] is None or result > previous["outcome"])):
                entries[key] = {"key": key, "history": history, "move": move, "outcome": result}
    return entries


def write_book(path, entries):
    rows = sorted(entries.values(), key=lambda row: (len(row["history"]), row["history"]))
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
        output.write("[\n")
        output.write(",\n".join("  " + json.dumps(row) for row in rows))
        output.write("\n]\n")
    Path(output.name).replace(path)


def init_worker(path):
    global LIBRARY
    LIBRARY = _bind_native(path)


def search_opening(task):
    position, scale = task
    current, occupied, history = position
    played = len(history)
    if played == 0:
        return 3, 1

    legal = GEOMETRY.landings(occupied)
    wins = legal & GEOMETRY.wins(current, occupied)
    if wins:
        return GEOMETRY.first(wins), 1
    safe = _safe_moves(legal, GEOMETRY.wins(current ^ occupied, occupied))
    if not safe:
        return GEOMETRY.first(legal), -1

    if played <= 2:
        search_time, proof_time = 2.1, 5.0
    elif played <= 4:
        search_time, proof_time = 0.38, 1.15
    elif played <= 6:
        search_time, proof_time = 0.10, 0.45
    else:
        search_time, proof_time = 0.04, 0.16

    move = LIBRARY.cx_choose(current, occupied, search_time * scale, None, None)
    proven = ctypes.c_int(-1)
    outcome = LIBRARY.cx_prove(current, occupied, proof_time * scale, move, ctypes.byref(proven), None)
    if outcome in (0, 1):
        move = proven.value
    return move, outcome if outcome != UNKNOWN else None


def expand(layer, decisions):
    following = {}
    for (position, seats), (chosen, _) in zip(layer.values(), decisions):
        current, occupied, history = position
        seat = 1 << (len(history) % 2)
        legal = GEOMETRY.landings(occupied)
        wins = legal & GEOMETRY.wins(current, occupied)
        for column in GEOMETRY.order:
            move = legal & GEOMETRY.column_masks[column]
            child_seats = seats if column == chosen else seats & (3 ^ seat)
            if not child_seats or not move or move & wins or occupied | move == GEOMETRY.board:
                continue
            key, child, _ = canonical(current ^ occupied, occupied | move, history + str(column))
            if key in following:
                old, old_seats = following[key]
                following[key] = old, old_seats | child_seats
            else:
                following[key] = child, child_seats
    return following


def prove_first_reply(task):
    (current, occupied, history), scale = task
    budget = 2.0 if len(history) <= 4 else 0.7 if len(history) <= 8 else 0.17
    preferred = LIBRARY.cx_choose(current, occupied, min(0.32, budget) * scale, None, None)
    column = ctypes.c_int(-1)
    result = LIBRARY.cx_can_force(current, occupied, budget * scale, 1, preferred, ctypes.byref(column))
    if result == 1:
        return column.value
    if result == 0:
        raise RuntimeError(f"A previously winning strategy reached {history} without a winning reply")

    safe = _safe_moves(GEOMETRY.landings(occupied), GEOMETRY.wins(current ^ occupied, occupied))
    for move in [preferred, *(c for c in GEOMETRY.order if c != preferred)]:
        landing = safe & GEOMETRY.column_masks[move]
        if landing and LIBRARY.cx_can_force(
                current ^ occupied, occupied | landing, budget * scale / 3, 0, -1, ctypes.byref(column)) == 0:
            return move
    return None


def cover_first_player(pool, entries, last_ply, scale):
    layer = {0: ((0, 0, ""), 1)}
    for ply in range(last_ply + 1):
        if ply % 2:
            layer = expand(layer, [(-1, None)] * len(layer))
            continue
        moves = {}
        missing = {}
        for key, (position, _) in layer.items():
            current, occupied, _ = position
            legal = GEOMETRY.landings(occupied)
            safe = _safe_moves(legal, GEOMETRY.wins(current ^ occupied, occupied))
            if not ply:
                moves[key] = 3
            elif legal & GEOMETRY.wins(current, occupied):
                continue
            elif key in entries and entries[key]["outcome"] == 1:
                moves[key] = entries[key]["move"]
            elif safe and not safe & (safe - 1):
                moves[key] = GEOMETRY.first(safe)
            else:
                missing[key] = (position, scale)

        results = pool.map(prove_first_reply, missing.values(), chunksize=2)
        unresolved = 0
        for key, move in zip(missing, results):
            if move is None:
                unresolved += 1
                continue
            history = layer[key][0][2]
            entries[key] = {"key": key, "history": history, "move": move, "outcome": 1}
            moves[key] = move
        print(f"first-player ply {ply:2}: {len(layer):5} positions, {unresolved:5} unresolved", flush=True)
        layer = expand({key: layer[key] for key in moves}, [(move, 1) for move in moves.values()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plies", type=int, default=9, help="largest number of stones in a book position")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--scale", type=float, default=1.0, help="multiplier for time per position")
    parser.add_argument("--seed", type=Path, action="append", default=[])
    parser.add_argument("--first-win-plies", type=int, default=0, help="extend the proven first-player strategy")
    parser.add_argument("--output", type=Path, default=ROOT / "openings.json")
    args = parser.parse_args()
    if (not 0 <= args.plies < GEOMETRY.size or not 0 <= args.first_win_plies < GEOMETRY.size
            or args.jobs < 1 or not math.isfinite(args.scale) or args.scale <= 0):
        parser.error("expected plies between 0 and 41, at least one job, and finite positive scale")

    rows = read_opening_rows(args.seed)
    seeds = {key: (row["move"], row["outcome"]) for key, row in rows.items()}
    layer = {0: ((0, 0, ""), 3)}
    saved = {key: row for key, row in rows.items() if row["outcome"] in (0, 1) or len(row["history"]) == 1}
    with tempfile.TemporaryDirectory(prefix="connectx-book-") as directory:
        path = Path(directory) / "engine-book.so"
        build_host(path, ["-DCX_PROOF_BITS=24", "-DCX_SEARCH_BITS=21"])
        with ProcessPoolExecutor(args.jobs, initializer=init_worker, initargs=(path,)) as pool:
            for ply in range(args.plies + 1):
                tasks = [(position, args.scale) for key, (position, _) in layer.items() if key not in seeds]
                results = iter(pool.map(search_opening, tasks, chunksize=2))
                decisions = [seeds[key] if key in seeds else next(results) for key in layer]
                proven = 0
                for (key, (position, _)), (move, outcome) in zip(layer.items(), decisions):
                    if not ply:
                        continue
                    if outcome in (0, 1):
                        proven += 1
                    elif ply != 1:
                        continue
                    saved[key] = {"key": key, "history": position[2], "move": move, "outcome": outcome}
                print(f"ply {ply:2}: {len(layer):5} positions, {proven:5} proven nonlosing moves, {len(saved):5} saved", flush=True)
                if ply < args.plies:
                    layer = expand(layer, decisions)
            if args.first_win_plies:
                cover_first_player(pool, saved, args.first_win_plies, args.scale)
    write_book(args.output, saved)


if __name__ == "__main__":
    main()
