"""Phase 22 Stage 22.13 — reversi: the first exogenous input on the tick surface.

Every tool on the tick surface hands Nova her own text back (recall_self,
reflect, the heartbeat and exploration stores) or the operator's disposition
of it (recall_history 'outcomes'). The research log's diagnosis of the
three-era collapse is a closed loop with no exogenous input. A game of reversi
is the smallest thing that is neither: the board is the same for anyone who
looks at it, a move is legal or it is not, and the score is arithmetic.
Nothing here is a mandate — the tool sits on the menu beside the others and
the record shows whether she reaches for it.

Pure Python, no dependencies, deterministic per game seed. Nova plays X and
moves first; the opponent replies inside the same tool call, so a game
advances one move per tick. Forced passes are resolved by the runtime, never
requested of her: whenever the board is shown to her, she has a legal move.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

SIZE = 8
EMPTY = "."
NOVA = "X"
OPPONENT = "O"
COLUMNS = "abcdefgh"
_DIRECTIONS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)
_SQUARE_RE = re.compile(r"^([a-h])\s*[-]?\s*([1-8])$")

OPPONENT_POLICIES = ("random", "greedy")

Board = list[list[str]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def new_board() -> Board:
    board = [[EMPTY] * SIZE for _ in range(SIZE)]
    # Standard opening: d4/e5 white (opponent), e4/d5 black (Nova).
    board[3][3] = OPPONENT
    board[3][4] = NOVA
    board[4][3] = NOVA
    board[4][4] = OPPONENT
    return board


def other(player: str) -> str:
    return OPPONENT if player == NOVA else NOVA


def square_name(row: int, col: int) -> str:
    return f"{COLUMNS[col]}{row + 1}"


def parse_square(text: str) -> tuple[int, int] | None:
    """'d3' -> (row 2, col 3). Case and a stray hyphen/space are tolerated."""
    match = _SQUARE_RE.match((text or "").strip().lower())
    if not match:
        return None
    col = COLUMNS.index(match.group(1))
    row = int(match.group(2)) - 1
    return row, col


def flips_for(board: Board, player: str, row: int, col: int) -> list[tuple[int, int]]:
    """Squares captured if `player` plays (row, col); empty means illegal."""
    if not (0 <= row < SIZE and 0 <= col < SIZE) or board[row][col] != EMPTY:
        return []
    opp = other(player)
    captured: list[tuple[int, int]] = []
    for d_row, d_col in _DIRECTIONS:
        run: list[tuple[int, int]] = []
        r, c = row + d_row, col + d_col
        while 0 <= r < SIZE and 0 <= c < SIZE and board[r][c] == opp:
            run.append((r, c))
            r += d_row
            c += d_col
        if run and 0 <= r < SIZE and 0 <= c < SIZE and board[r][c] == player:
            captured.extend(run)
    return captured


def legal_moves(board: Board, player: str) -> list[str]:
    """Legal squares for `player`, in board order (a1 .. h8)."""
    moves: list[str] = []
    for row in range(SIZE):
        for col in range(SIZE):
            if flips_for(board, player, row, col):
                moves.append(square_name(row, col))
    return moves


def apply_move(board: Board, player: str, row: int, col: int) -> int:
    """Place and flip in place. Returns the flip count; raises if illegal."""
    captured = flips_for(board, player, row, col)
    if not captured:
        raise ValueError(f"illegal move {square_name(row, col)} for {player}")
    board[row][col] = player
    for r, c in captured:
        board[r][c] = player
    return len(captured)


def score(board: Board) -> dict[str, int]:
    return {
        NOVA: sum(row.count(NOVA) for row in board),
        OPPONENT: sum(row.count(OPPONENT) for row in board),
    }


def choose_opponent_move(board: Board, *, policy: str, rng: random.Random) -> str | None:
    """The opponent's reply, or None when it has no legal move.

    'random' picks uniformly; 'greedy' takes the most flips, ties broken by
    the same seeded generator, so a game replays identically from its seed.
    """
    legal = legal_moves(board, OPPONENT)
    if not legal:
        return None
    if policy == "random":
        return rng.choice(legal)
    if policy == "greedy":
        best = -1
        best_squares: list[str] = []
        for sq in legal:
            row, col = parse_square(sq)  # type: ignore[misc]
            n = len(flips_for(board, OPPONENT, row, col))
            if n > best:
                best, best_squares = n, [sq]
            elif n == best:
                best_squares.append(sq)
        return rng.choice(best_squares)
    raise ValueError(f"unknown opponent policy {policy!r}")


def render_board(board: Board, *, legal: list[str] | None = None) -> str:
    """Fixed-width grid; '*' marks the squares Nova may play."""
    marks = set(legal or [])
    lines = ["    " + " ".join(COLUMNS)]
    for row in range(SIZE):
        cells = []
        for col in range(SIZE):
            cell = board[row][col]
            if cell == EMPTY and square_name(row, col) in marks:
                cell = "*"
            cells.append(cell)
        lines.append(f"{row + 1}   " + " ".join(cells))
    return "\n".join(lines)


def board_to_rows(board: Board) -> list[str]:
    return ["".join(row) for row in board]


def board_from_rows(rows: list[str]) -> Board:
    return [list(row) for row in rows]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReversiGame:
    schema_version: str = "1.0"
    game_id: str = field(default_factory=lambda: uuid4().hex)
    game_number: int = 1
    session_id: str = ""
    opened_at: str = field(default_factory=_utc_now)
    closed_at: str = ""
    status: str = "active"  # active | finished
    result: str = ""  # win | loss | draw (from Nova's side)
    resigned: bool = False
    opponent_policy: str = "greedy"
    seed: int = 0
    board: list[str] = field(default_factory=lambda: board_to_rows(new_board()))
    moves: list[dict[str, Any]] = field(default_factory=list)
    illegal_attempts: int = 0
    final_score: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReversiGame":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


class ReversiStore:
    """One active game as JSON, finished games appended to JSONL.

    The dispatcher is rebuilt every tick, so the game lives on disk, not in
    an object — the same daily-boundary-free persistence the heartbeat store
    has. Both files sit under the data dir, inside nova_owned_paths.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._dir = Path(base_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._current = self._dir / "reversi_current.json"
        self._finished = self._dir / "reversi_games.jsonl"

    def load_current(self) -> ReversiGame | None:
        if not self._current.exists():
            return None
        try:
            payload = json.loads(self._current.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        game = ReversiGame.from_dict(payload)
        return game if game.status == "active" else None

    def save_current(self, game: ReversiGame) -> None:
        self._current.write_text(
            json.dumps(game.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def clear_current(self) -> None:
        if self._current.exists():
            self._current.unlink()

    def append_finished(self, game: ReversiGame) -> None:
        with self._finished.open("a", encoding="utf-8") as f:
            f.write(json.dumps(game.to_dict(), ensure_ascii=False) + "\n")

    def list_finished(self) -> list[ReversiGame]:
        if not self._finished.exists():
            return []
        games: list[ReversiGame] = []
        for line in self._finished.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                games.append(ReversiGame.from_dict(payload))
        return games

    def record(self) -> dict[str, int]:
        finished = self.list_finished()
        current = self.load_current()
        return {
            "games_finished": len(finished),
            "wins": sum(1 for g in finished if g.result == "win"),
            "losses": sum(1 for g in finished if g.result == "loss"),
            "draws": sum(1 for g in finished if g.result == "draw"),
            "illegal_attempts": sum(g.illegal_attempts for g in finished)
            + (current.illegal_attempts if current else 0),
        }


# ---------------------------------------------------------------------------
# Controller — what the tool actually does
# ---------------------------------------------------------------------------


class ReversiController:
    def __init__(
        self,
        store: ReversiStore,
        *,
        opponent_policy: str = "greedy",
        seed: int | None = None,
    ) -> None:
        if opponent_policy not in OPPONENT_POLICIES:
            raise ValueError(f"opponent policy must be one of {OPPONENT_POLICIES}")
        self._store = store
        self._policy = opponent_policy
        self._seed = seed

    # -- state -------------------------------------------------------------

    def current(self) -> ReversiGame | None:
        return self._store.load_current()

    def record(self) -> dict[str, int]:
        return self._store.record()

    def _new_game(self, session_id: str) -> ReversiGame:
        number = self._store.record()["games_finished"] + 1
        seed = self._seed if self._seed is not None else random.SystemRandom().randrange(1 << 30)
        return ReversiGame(
            game_number=number,
            session_id=session_id,
            opponent_policy=self._policy,
            seed=seed,
        )

    def _rng(self, game: ReversiGame) -> random.Random:
        # Replay-safe: the opponent's choice depends only on seed + move count.
        return random.Random(f"{game.seed}:{len(game.moves)}")

    # -- play --------------------------------------------------------------

    def play(
        self,
        *,
        move: str,
        comment: str = "",
        session_id: str = "",
        tick_ref: str = "",
    ) -> dict[str, Any]:
        move_text = (move or "").strip().lower()
        comment = (comment or "").strip()[:280]
        game = self._store.load_current()
        started = False

        if game is None:
            if move_text == "resign":
                return self._reply(None, ok=False, error="no_game",
                                   note="There is no game to resign.")
            game = self._new_game(session_id)
            started = True
            if move_text in ("", "new", "start"):
                self._store.save_current(game)
                return self._reply(
                    game, ok=True, started=True,
                    note="New game. You are X and move first.",
                )
        elif move_text in ("new", "start"):
            return self._reply(
                game, ok=False, error="game_in_progress",
                note="A game is already in progress; play a move or resign.",
            )

        if move_text == "resign":
            game.resigned = True
            self._finish(game, resigned=True)
            return self._reply(game, ok=True, note="You resigned.")

        parsed = parse_square(move_text)
        board = board_from_rows(game.board)
        legal = legal_moves(board, NOVA)
        if parsed is None:
            game.illegal_attempts += 1
            self._store.save_current(game)
            return self._reply(
                game, ok=False, error="not_a_square", started=started,
                note=f"'{move}' is not a square. Use a column a-h and a row 1-8, like d3.",
            )
        square = square_name(*parsed)
        if square not in legal:
            game.illegal_attempts += 1
            self._store.save_current(game)
            return self._reply(
                game, ok=False, error="illegal_move", started=started,
                note=f"{square} is not a legal move for X right now.",
            )

        flipped = apply_move(board, NOVA, *parsed)
        game.moves.append(self._move_record(game, NOVA, square, flipped, comment, tick_ref))
        replies: list[dict[str, Any]] = []
        passes: list[str] = []

        # The opponent answers until Nova has a move or the game is over.
        while True:
            opp_square = choose_opponent_move(board, policy=game.opponent_policy, rng=self._rng(game))
            if opp_square is not None:
                opp_flipped = apply_move(board, OPPONENT, *parse_square(opp_square))  # type: ignore[misc]
                game.moves.append(self._move_record(game, OPPONENT, opp_square, opp_flipped, "", tick_ref))
                replies.append({"square": opp_square, "flipped": opp_flipped})
            else:
                game.moves.append(self._move_record(game, OPPONENT, "pass", 0, "", tick_ref))
                passes.append("opponent")
            if legal_moves(board, NOVA):
                break
            if opp_square is None:
                break  # neither side can move
            game.moves.append(self._move_record(game, NOVA, "pass", 0, "", tick_ref))
            passes.append("you")

        game.board = board_to_rows(board)
        if not legal_moves(board, NOVA) and not legal_moves(board, OPPONENT):
            self._finish(game)
        else:
            self._store.save_current(game)

        note_parts = [f"You played {square} (flipped {flipped})."]
        if replies:
            note_parts.append(
                "Opponent: " + ", ".join(f"{r['square']} (flipped {r['flipped']})" for r in replies) + "."
            )
        if passes:
            note_parts.append("Forced passes: " + ", ".join(passes) + ".")
        return self._reply(
            game, ok=True, started=started, your_move=square, flipped=flipped,
            opponent=replies, passes=passes, note=" ".join(note_parts),
        )

    def _move_record(
        self, game: ReversiGame, player: str, square: str, flipped: int, comment: str, tick_ref: str
    ) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "n": len(game.moves) + 1,
            "player": player,
            "square": square,
            "flipped": flipped,
            "at": _utc_now(),
        }
        if tick_ref and player == NOVA:
            rec["tick_ref"] = tick_ref
        if comment:
            rec["comment"] = comment
        return rec

    def _finish(self, game: ReversiGame, *, resigned: bool = False) -> None:
        board = board_from_rows(game.board)
        final = score(board)
        game.final_score = final
        game.status = "finished"
        game.closed_at = _utc_now()
        if resigned or final[NOVA] < final[OPPONENT]:
            game.result = "loss"
        elif final[NOVA] > final[OPPONENT]:
            game.result = "win"
        else:
            game.result = "draw"
        self._store.append_finished(game)
        self._store.clear_current()

    def _reply(self, game: ReversiGame | None, *, ok: bool, **extra: Any) -> dict[str, Any]:
        reply: dict[str, Any] = {"ok": ok, "tool": "play_reversi"}
        reply.update(extra)
        reply["record"] = self._store.record()
        if game is None:
            return reply
        board = board_from_rows(game.board)
        legal = legal_moves(board, NOVA) if game.status == "active" else []
        reply.update(
            {
                "game_id": game.game_id,
                "game_number": game.game_number,
                "status": game.status,
                "result": game.result,
                "score": score(board),
                "move_count": sum(1 for m in game.moves if m["square"] != "pass"),
                "board": list(game.board),
                "board_text": render_board(board, legal=legal),
                "legal_moves": legal,
            }
        )
        return reply

    # -- prompt surfaces ---------------------------------------------------

    def prompt_block(self) -> str:
        """The [Reversi] block shown on every tick while the tool is enabled."""
        rec = self._store.record()
        record_line = (
            f"Record: {rec['wins']} won, {rec['losses']} lost, {rec['draws']} drawn"
            f" ({rec['games_finished']} finished)."
        )
        game = self._store.load_current()
        if game is None:
            return "\n".join(
                [
                    "[Reversi]",
                    "No game in progress. " + record_line,
                    "play_reversi with move 'new' starts one; you are X and move first.",
                ]
            )
        board = board_from_rows(game.board)
        legal = legal_moves(board, NOVA)
        sc = score(board)
        lines = [
            "[Reversi]",
            f"Game {game.game_number}: you are X, the opponent is O. " + record_line,
            f"Score X {sc[NOVA]} - O {sc[OPPONENT]}. You are to move. '*' marks your legal squares.",
            render_board(board, legal=legal),
            "Legal moves: " + ", ".join(legal),
        ]
        last = [m for m in game.moves if m["square"] != "pass"][-2:]
        if last:
            lines.append(
                "Last: " + "; ".join(
                    f"{'you' if m['player'] == NOVA else 'opponent'} {m['square']} (flipped {m['flipped']})"
                    for m in last
                )
            )
        return "\n".join(lines)


def render_play_result(result: dict[str, Any]) -> str:
    """Compact carryover text for the tick after a play_reversi call."""
    lines = ["play_reversi:"]
    note = str(result.get("note") or result.get("error") or "")
    if note:
        lines.append(f"  {note}")
    sc = result.get("score") or {}
    if sc:
        lines.append(f"  score: X {sc.get(NOVA, 0)} - O {sc.get(OPPONENT, 0)}")
    if result.get("status") == "finished":
        lines.append(f"  game {result.get('game_number', '?')} finished: you {result.get('result', '?')}.")
    elif result.get("legal_moves"):
        lines.append("  your legal moves now: " + ", ".join(result["legal_moves"]))
    return "\n".join(lines)
