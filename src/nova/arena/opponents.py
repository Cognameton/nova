"""Scripted Reversi players for the Phase 23 opponent pool.

Classic computer-player style: rules plus search, nothing learned. Every
player is colour-agnostic (it is told which side it plays), deterministic
given the rng it is handed, and never returns an illegal move.

Styles:
  random     uniform legal move
  greedy     most flips; ties by rng.choice over the tied squares, the same
             draw nova.agent.reversi makes, so an arena greedy O replays a
             live game exactly
  corner     static corner/edge weight table (the ~81%-vs-greedy player)
  mobility   minimise the opponent's legal-move count after the move
  lookahead  alpha-beta minimax on the weight table, `depth` plies

Strength dial: epsilon, the chance of a uniform random legal move instead
of the style's choice. A player is written as a spec string,
"style[:epsilon][:depth]", e.g. "corner:0.25" or "lookahead:0:3".
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from nova.agent.reversi import (
    SIZE,
    Board,
    apply_move,
    flips_for,
    legal_moves,
    other,
    parse_square,
    score,
)

STYLES = ("random", "greedy", "corner", "mobility", "lookahead")

# Phase 23 plan: the mobility style is never used for training; mobility at
# epsilon 0.1 is the evaluation-only transfer opponent.
HELD_OUT_STYLE = "mobility"
HELD_OUT_SPEC = "mobility:0.1"
EVAL_SPEC = "greedy"

_QUADRANT = (
    (100, -20, 10, 5),
    (-20, -50, -2, -2),
    (10, -2, 1, 1),
    (5, -2, 1, 0),
)
# Full 8x8 table by mirroring the top-left quadrant.
WEIGHTS: tuple[tuple[int, ...], ...] = tuple(
    tuple(_QUADRANT[min(r, SIZE - 1 - r)][min(c, SIZE - 1 - c)] for c in range(SIZE))
    for r in range(SIZE)
)


class Player(Protocol):
    name: str

    def choose(self, board: Board, side: str, legal: list[str], rng: random.Random) -> str:
        ...


def _copy(board: Board) -> Board:
    return [row[:] for row in board]


def _weight(square: str) -> int:
    row, col = parse_square(square)  # type: ignore[misc]
    return WEIGHTS[row][col]


def positional_eval(board: Board, side: str) -> int:
    """Weight-table sum for `side` minus the same for its opponent."""
    opp = other(side)
    total = 0
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] == side:
                total += WEIGHTS[r][c]
            elif board[r][c] == opp:
                total -= WEIGHTS[r][c]
    return total


def _after(board: Board, side: str, square: str) -> Board:
    nxt = _copy(board)
    apply_move(nxt, side, *parse_square(square))  # type: ignore[misc]
    return nxt


def _pick_best(scored: list[tuple[float, str]], rng: random.Random) -> str:
    best = max(s for s, _ in scored)
    return rng.choice([sq for s, sq in scored if s == best])


def _greedy(board: Board, side: str, legal: list[str], rng: random.Random) -> str:
    # Mirrors nova.agent.reversi.choose_opponent_move: legal order, then
    # rng.choice over the tied squares.
    best = -1
    best_squares: list[str] = []
    for sq in legal:
        n = len(flips_for(board, side, *parse_square(sq)))  # type: ignore[misc]
        if n > best:
            best, best_squares = n, [sq]
        elif n == best:
            best_squares.append(sq)
    return rng.choice(best_squares)


def _corner(board: Board, side: str, legal: list[str], rng: random.Random) -> str:
    return _pick_best([(_weight(sq), sq) for sq in legal], rng)


def _mobility(board: Board, side: str, legal: list[str], rng: random.Random) -> str:
    opp = other(side)
    return _pick_best([(-len(legal_moves(_after(board, side, sq), opp)), sq) for sq in legal], rng)


def _terminal_value(board: Board, side: str) -> int:
    s = score(board)
    diff = s[side] - s[other(side)]
    return 10_000 * (diff > 0) - 10_000 * (diff < 0)


def _minimax(board: Board, to_move: str, me: str, depth: int, alpha: float, beta: float) -> float:
    legal = legal_moves(board, to_move)
    if not legal:
        if not legal_moves(board, other(to_move)):
            return _terminal_value(board, me)
        return _minimax(board, other(to_move), me, depth, alpha, beta)
    if depth == 0:
        return positional_eval(board, me)
    if to_move == me:
        value = float("-inf")
        for sq in legal:
            value = max(value, _minimax(_after(board, to_move, sq), other(to_move), me, depth - 1, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    value = float("inf")
    for sq in legal:
        value = min(value, _minimax(_after(board, to_move, sq), other(to_move), me, depth - 1, alpha, beta))
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value


@dataclass(slots=True)
class ScriptedPlayer:
    style: str
    epsilon: float = 0.0
    depth: int = 2

    def __post_init__(self) -> None:
        if self.style not in STYLES:
            raise ValueError(f"unknown style {self.style!r}; choose from {STYLES}")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if self.depth < 1:
            raise ValueError("depth must be >= 1")

    @property
    def name(self) -> str:
        return spec_of(self)

    def choose(self, board: Board, side: str, legal: list[str], rng: random.Random) -> str:
        if not legal:
            raise ValueError("choose() called with no legal moves")
        # Epsilon draw first, and only when epsilon > 0, so an epsilon-0
        # player consumes the rng exactly like the live greedy opponent.
        if self.epsilon > 0 and rng.random() < self.epsilon:
            return rng.choice(legal)
        if self.style == "random":
            return rng.choice(legal)
        if self.style == "greedy":
            return _greedy(board, side, legal, rng)
        if self.style == "corner":
            return _corner(board, side, legal, rng)
        if self.style == "mobility":
            return _mobility(board, side, legal, rng)
        scored = [
            (_minimax(_after(board, side, sq), other(side), side, self.depth - 1,
                      float("-inf"), float("inf")), sq)
            for sq in legal
        ]
        return _pick_best(scored, rng)


def spec_of(player: ScriptedPlayer) -> str:
    parts = [player.style]
    if player.epsilon or player.style == "lookahead":
        parts.append(f"{player.epsilon:g}")
    if player.style == "lookahead":
        parts.append(str(player.depth))
    return ":".join(parts)


def parse_spec(spec: str) -> ScriptedPlayer:
    """'corner:0.25' -> ScriptedPlayer('corner', 0.25); 'lookahead:0:3' sets depth."""
    parts = [p.strip() for p in spec.split(":")]
    style = parts[0]
    epsilon = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
    depth = int(parts[2]) if len(parts) > 2 and parts[2] else 2
    return ScriptedPlayer(style, epsilon, depth)


def training_pool(epsilons: tuple[float, ...] = (0.0, 0.25, 0.5)) -> list[ScriptedPlayer]:
    """Every style except the held-out one, at each epsilon (random once)."""
    return [
        ScriptedPlayer(s, e)
        for s in STYLES if s != HELD_OUT_STYLE
        for e in (epsilons if s != "random" else (0.0,))
    ]
