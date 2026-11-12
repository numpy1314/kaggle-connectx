"""Check move correctness and play paired local games with the submission."""

import argparse
import ctypes
import json
import random
import statistics
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STANDARD = (6, 7, 4)


def load_agent(path, *, book=True, native=True):
    # The Kaggle .py loader executes source in a new namespace and uses its last callable.
    scope = {}
    exec(compile(path.read_bytes(), str(path), "exec"), scope)
    if not book:
        scope["_OPENING_BYTES"] = b""
    if not native:
        scope["_NATIVE_BLOBS"] = {}
    return [value for value in scope.values() if callable(value)][-1], scope


class Board:
    def __init__(self, dimensions=STANDARD):
        self.rows, self.columns, self.target = dimensions
        self.initial = (0,) * (self.rows * self.columns)

    def legal(self, board):
        return [column for column in range(self.columns) if board[column] == 0]

    def place(self, board, column, mark):
        if type(column) is not int or not 0 <= column < self.columns or board[column]:
            raise AssertionError(f"Illegal column {column!r}")
        row = next(row for row in reversed(range(self.rows)) if not board[row * self.columns + column])
        index = row * self.columns + column
        child = board[:index] + (mark,) + board[index + 1:]
        won = any(
            self.line_length(child, row, column, mark, dr, dc) >= self.target
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
        )
        return child, won

    def line_length(self, board, row, column, mark, dr, dc):
        length = 1
        for sign in (-1, 1):
            r, c = row + sign * dr, column + sign * dc
            while 0 <= r < self.rows and 0 <= c < self.columns and board[r * self.columns + c] == mark:
                length += 1
                r += sign * dr
                c += sign * dc
        return length

    def replay(self, history):
        board = self.initial
        for i, column in enumerate(history):
            board, won = self.place(board, int(column), 1 + i % 2)
            if won:
                raise ValueError(f"Prefix contains a completed game: {history}")
        return board

    def prefix(self, rng, plies):
        while True:
            board = self.initial
            history = []
            for i in range(plies):
                columns = self.legal(board)
                rng.shuffle(columns)
                for column in columns:
                    child, won = self.place(board, column, 1 + i % 2)
                    if not won:
                        history.append(str(column))
                        board = child
                        break
                else:
                    break
            if len(history) == plies:
                return "".join(history), board

    def configuration(self, timeout):
        return {"rows": self.rows, "columns": self.columns, "inarow": self.target, "actTimeout": timeout}


def first_player_coverage(geometry, rows, safe_moves):
    proven = {row["key"]: row["move"] for row in rows if row["outcome"] == 1}
    last_ply = max(len(row["history"]) for row in rows)
    positions = {(0, 0)}
    coverage = {}
    for ply in range(last_ply + 1):
        following = set()
        missing = 0
        for current, occupied in positions:
            legal = geometry.landings(occupied)
            wins = legal & geometry.wins(current, occupied)
            if not ply:
                legal &= geometry.column_masks[3]
            elif not ply % 2:
                if wins:
                    continue
                safe = safe_moves(legal, geometry.wins(current ^ occupied, occupied))
                key = current + occupied
                if not safe:
                    raise AssertionError(f"First-player proof reached a forced loss at key {key}")
                if not safe & (safe - 1):
                    legal = safe
                elif key in proven:
                    legal = safe & geometry.column_masks[proven[key]]
                    if not legal:
                        raise AssertionError(f"Proven reply is unsafe at key {key}")
                else:
                    missing += 1
                    continue
            elif wins:
                raise AssertionError(f"First-player proof permits a second-player win at ply {ply}")

            while legal:
                move = legal & -legal
                legal ^= move
                other, occupied_after = current ^ occupied, occupied | move
                _, reflected = geometry.key(other, occupied_after)
                if reflected:
                    other, occupied_after = geometry.reflect(other), geometry.reflect(occupied_after)
                following.add((other, occupied_after))
        if not ply % 2:
            coverage[ply] = {"positions": len(positions), "unresolved": missing}
        positions = following
    return coverage


def verify(path, seed):
    agent, scope = load_agent(path)
    fallback, _ = load_agent(path, book=False, native=False)
    native = scope["_native_engine"]()
    if native is None:
        raise AssertionError("The expected native library did not load")

    rng = random.Random(seed)
    checked = 0
    for dimensions in (STANDARD, (4, 4, 3), (5, 8, 5), (3, 9, 4), (2, 6, 4), (1, 7, 4)):
        board = Board(dimensions)
        geometry = scope["_geometry"](*dimensions)
        for _ in range(120):
            played = rng.randrange(min(geometry.size, 30))
            _, state = board.prefix(rng, played)
            mark = 1 + played % 2
            current, occupied = geometry.encode(state, mark)
            legal = geometry.landings(occupied)
            for player, stones in ((mark, current), (3 - mark, current ^ occupied)):
                expected = [c for c in board.legal(state) if board.place(state, c, player)[1]]
                actual = [c for c in range(board.columns) if legal & geometry.wins(stones, occupied)
                          & geometry.column_masks[c]]
                if expected != actual:
                    raise AssertionError((dimensions, state, player, expected, actual))
            move = fallback({"board": state, "mark": mark}, board.configuration(0.001))
            if move not in board.legal(state):
                raise AssertionError((dimensions, state, move))
            checked += 1

    board = Board()

    @lru_cache(maxsize=200_000)
    def oracle(state, mark):
        value = -1
        for column in board.legal(state):
            child, won = board.place(state, column, mark)
            if won:
                return 1
            child_value = 0 if not board.legal(child) else -oracle(child, 3 - mark)
            value = max(value, child_value)
            if value == 1:
                break
        return value

    outcomes = Counter()
    for i in range(600):
        played = rng.randrange(29, 42)
        history, state = board.prefix(rng, played)
        mark = 1 + played % 2
        expected = oracle(state, mark)
        current, occupied = scope["_geometry"](*STANDARD).encode(state, mark)
        answer = ctypes.c_int(-1)
        value = native.cx_prove(current, occupied, 0.10, -1, ctypes.byref(answer), None)
        if value != expected:
            raise AssertionError((history, expected, value))
        played_column = agent({"board": state, "mark": mark}, board.configuration(0.20))
        for column in (played_column, answer.value) if value >= 0 else (played_column,):
            child, won = board.place(state, column, mark)
            result = 1 if won else 0 if not board.legal(child) else -oracle(child, 3 - mark)
            if result != expected:
                raise AssertionError((history, column, expected, result))
        outcomes[value] += 1

    geometry = scope["_geometry"](*STANDARD)
    book = scope["_opening_book"]()
    book_path = ROOT / "openings.json"
    coverage = {}
    if book and book_path.exists():
        rows = json.loads(book_path.read_text(encoding="utf-8"))
        if len(rows) != len(book):
            raise AssertionError("The submission has a different book size from openings.json")
        for row in rows:
            history = row["history"]
            state = board.replay(history)
            mark = 1 + len(history) % 2
            current, occupied = geometry.encode(state, mark)
            key, reflected = geometry.key(current, occupied)
            if reflected or key != row["key"] or book[key] != row["move"]:
                raise AssertionError(row)
            for candidate, column in ((state, row["move"]), (
                    board.replay("".join(str(6 - int(c)) for c in history)), 6 - row["move"])):
                child, won = board.place(candidate, column, mark)
                if row["outcome"] in (0, 1) and not won:
                    if any(board.place(child, reply, 3 - mark)[1] for reply in board.legal(child)):
                        raise AssertionError(row)
        coverage = first_player_coverage(geometry, rows, scope["_safe_moves"])

    print(json.dumps({
        "geometry_and_fallback_positions": checked,
        "endgames_matching_exhaustive_search": sum(outcomes.values()),
        "endgame_outcomes": {"win": outcomes[1], "draw": outcomes[0], "loss": outcomes[-1]},
        "opening_entries_checked": len(book),
        "first_player_coverage_by_plies": coverage,
    }, indent=2))


def play_games(path, opponent, pairs, timeout, seed):
    rng = random.Random(seed)
    board = Board()
    configuration = board.configuration(timeout)
    agent, _ = load_agent(path)
    if opponent == "random":
        def rival(observation, _configuration):
            return rng.choice(board.legal(observation["board"]))
    else:
        rival, _ = load_agent(path, book=False, native=opponent == "no-book")

    timings = []
    results = {"first": Counter(), "second": Counter()}
    fixed = ("", "0", "1", "2", "3", "30", "31", "32", "33", "3332", "3333", "3410")
    for index in range(pairs):
        history = fixed[index] if index < len(fixed) else board.prefix(rng, 4 + index % 3)[0]
        for seat in (1, 2):
            state = board.replay(history)
            mark = 1 + len(history) % 2
            winner = 0
            while board.legal(state):
                turn = agent if mark == seat else rival
                start = time.monotonic()
                column = turn({"board": state, "mark": mark}, configuration)
                elapsed = time.monotonic() - start
                if mark == seat:
                    timings.append(elapsed)
                    if elapsed >= timeout:
                        raise AssertionError(f"Timeout after {elapsed:.3f}s in game {index}, seat {seat}")
                state, won = board.place(state, column, mark)
                if won:
                    winner = mark
                    break
                mark = 3 - mark
            result = "win" if winner == seat else "loss" if winner else "draw"
            results["first" if seat == 1 else "second"][result] += 1
    report_results(results, timings, pairs * 2, opponent, timeout)


def report_results(results, timings, games, opponent, timeout):
    counts = sum(results.values(), Counter())
    print(json.dumps({
        "games": games,
        "opponent": opponent,
        "act_timeout_seconds": timeout,
        "win_draw_loss": [counts["win"], counts["draw"], counts["loss"]],
        "by_seat": {seat: [counts[key] for key in ("win", "draw", "loss")]
                    for seat, counts in results.items()},
        "timed_turns": len(timings),
        "median_seconds": round(statistics.median(timings), 6) if timings else None,
        "max_seconds": round(max(timings), 6) if timings else None,
    }, indent=2), flush=True)


def kaggle_games(path, pairs, timeout, seed):
    from kaggle_environments import make

    random.seed(seed)
    candidate = str(path.resolve())
    config = {"actTimeout": timeout}
    environment = make("connectx", configuration=config)
    final = environment.run([candidate, candidate])[-1]
    if [state.status for state in final] != ["DONE", "DONE"]:
        raise AssertionError(final)
    print("Kaggle self-play finished", [state.reward for state in final], flush=True)
    for opponent in ("random", "negamax"):
        results = {"first": Counter(), "second": Counter()}
        timings = []
        for _ in range(pairs):
            for seat in range(2):
                environment = make("connectx", configuration=config)
                agents = [opponent, opponent]
                agents[seat] = candidate
                final = environment.run(agents)[-1]
                if [state.status for state in final] != ["DONE", "DONE"]:
                    raise AssertionError(final)
                result = {1: "win", 0: "draw", -1: "loss"}[final[seat].reward]
                results["first" if seat == 0 else "second"][result] += 1
                for step in environment.logs:
                    if len(step) > seat and "duration" in step[seat]:
                        timings.append(step[seat]["duration"])
        if timings and max(timings) >= timeout:
            raise AssertionError(f"Kaggle measured a turn over {timeout} seconds")
        report_results(results, timings, pairs * 2, opponent, timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "play", "kaggle"))
    parser.add_argument("--file", type=Path, default=ROOT / "submission.py")
    parser.add_argument("--opponent", choices=("random", "no-book", "python"), default="no-book")
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    if args.pairs < 1 or args.timeout <= 0:
        parser.error("pairs and timeout must be positive")
    if args.mode == "verify":
        verify(args.file, args.seed)
    elif args.mode == "play":
        play_games(args.file, args.opponent, args.pairs, args.timeout, args.seed)
    else:
        kaggle_games(args.file, args.pairs, args.timeout, args.seed)


if __name__ == "__main__":
    main()
