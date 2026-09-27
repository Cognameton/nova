"""Tests for Phase 23 Stage 23.1 — the arena, the scripted opponent pool,
and the model player (with a fake generator; no GPU)."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from nova.agent.reversi import (
    NOVA,
    OPPONENT,
    ReversiController,
    ReversiStore,
    board_from_rows,
    legal_moves,
    new_board,
)
from nova.arena.calibrate import default_configs, fit_ratings
from nova.arena.harness import (
    CALIBRATION_SEED_RANGE,
    EVAL_SEEDS,
    TRAIN_SEED_RANGE,
    calibration_seeds,
    move_rng,
    play_game,
    run_match,
    summarize,
    train_seeds,
    wilson,
)
from nova.arena.model_player import (
    ModelPlayer,
    answer_json,
    build_messages,
    parse_answer,
)
from nova.arena.opponents import (
    HELD_OUT_SPEC,
    HELD_OUT_STYLE,
    STYLES,
    ScriptedPlayer,
    parse_spec,
    spec_of,
    training_pool,
)


class SeedRangeTests(unittest.TestCase):
    def test_ranges_are_disjoint(self) -> None:
        sets = [set(EVAL_SEEDS), set(TRAIN_SEED_RANGE), set(CALIBRATION_SEED_RANGE)]
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(sets[i] & sets[j])

    def test_seed_helpers_refuse_to_leave_their_range(self) -> None:
        self.assertEqual(train_seeds(3)[0], TRAIN_SEED_RANGE.start)
        with self.assertRaises(ValueError):
            train_seeds(10, offset=len(TRAIN_SEED_RANGE) - 5)
        with self.assertRaises(ValueError):
            calibration_seeds(-1)

    def test_eval_is_300_games(self) -> None:
        self.assertEqual(len(EVAL_SEEDS), 300)


class OpponentTests(unittest.TestCase):
    def test_every_style_only_plays_legal_moves(self) -> None:
        players = [ScriptedPlayer(s, e) for s in STYLES for e in (0.0, 0.5)]
        players.append(ScriptedPlayer("lookahead", 0.0, 3))
        for p in players:
            for seed in range(3):
                # play_game raises on any illegal move.
                game = play_game(p, ScriptedPlayer("random"), seed)
                self.assertIn(game["result"], ("win", "loss", "draw"))
                game = play_game(ScriptedPlayer("random"), p, seed)
                self.assertEqual(sum(game["final_score"].values()) <= 64, True)

    def test_players_work_as_either_colour(self) -> None:
        board = new_board()
        for side in (NOVA, OPPONENT):
            legal = legal_moves(board, side)
            for s in STYLES:
                self.assertIn(ScriptedPlayer(s).choose(board, side, legal, random.Random(1)), legal)

    def test_corner_takes_an_available_corner(self) -> None:
        rows = [
            ".OX.....",
            "........",
            "........",
            "...OX...",
            "...XO...",
            "........",
            "........",
            "........",
        ]
        board = board_from_rows(rows)
        legal = legal_moves(board, NOVA)
        self.assertIn("a1", legal)
        self.assertEqual(ScriptedPlayer("corner").choose(board, NOVA, legal, random.Random(0)), "a1")
        self.assertEqual(ScriptedPlayer("lookahead").choose(board, NOVA, legal, random.Random(0)), "a1")

    def test_spec_round_trip(self) -> None:
        for spec in ("random", "greedy", "corner:0.25", "mobility:0.1", "lookahead:0:3", "lookahead:0.5:2"):
            self.assertEqual(spec_of(parse_spec(spec)), spec)
        with self.assertRaises(ValueError):
            parse_spec("bogus")
        with self.assertRaises(ValueError):
            parse_spec("greedy:1.5")

    def test_training_pool_excludes_held_out_style(self) -> None:
        pool = training_pool()
        self.assertTrue(pool)
        self.assertNotIn(HELD_OUT_STYLE, {p.style for p in pool})
        self.assertEqual(parse_spec(HELD_OUT_SPEC).style, HELD_OUT_STYLE)
        self.assertEqual(sum(p.style == "random" for p in pool), 1)


class HarnessTests(unittest.TestCase):
    def test_game_replays_identically_from_seed(self) -> None:
        a = play_game(parse_spec("corner:0.25"), parse_spec("greedy"), 7)
        b = play_game(parse_spec("corner:0.25"), parse_spec("greedy"), 7)
        self.assertEqual(a["moves"], b["moves"])
        c = play_game(parse_spec("corner:0.25"), parse_spec("greedy"), 8)
        self.assertNotEqual(a["moves"], c["moves"])

    def test_greedy_opponent_matches_the_live_runtime(self) -> None:
        # Same seed + same X moves => the live controller's greedy O and the
        # arena's greedy O make identical replies, passes included.
        x_player = parse_spec("corner:0.25")
        for seed in (3, 11, 42):
            with tempfile.TemporaryDirectory() as tmp:
                store = ReversiStore(tmp)
                ctl = ReversiController(store, opponent_policy="greedy", seed=seed)
                ctl.play(move="new")
                while (game := store.load_current()) is not None:
                    board = board_from_rows(game.board)
                    legal = legal_moves(board, NOVA)
                    sq = x_player.choose(board, NOVA, legal, move_rng(seed, len(game.moves), NOVA))
                    ctl.play(move=sq)
                live = store.list_finished()[-1]
            arena = play_game(x_player, parse_spec("greedy"), seed)
            strip = lambda ms: [(m["player"], m["square"]) for m in ms if m["square"] != "pass"]
            self.assertEqual(strip(live.moves), strip(arena["moves"]))
            self.assertEqual(live.final_score, arena["final_score"])

    def test_x_moves_record_the_board_they_saw(self) -> None:
        game = play_game(parse_spec("greedy"), parse_spec("random"), 1)
        first = game["moves"][0]
        self.assertEqual(first["player"], NOVA)
        self.assertEqual(board_from_rows(first["board_before"]), new_board())
        self.assertIn(first["square"], first["legal"])
        self.assertNotIn("board_before", game["moves"][1])

    def test_run_match_streams_records_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_match(parse_spec("corner"), parse_spec("random"), range(5),
                               out_dir=Path(tmp), run_id="t")
            lines = (Path(tmp) / "t" / "games.jsonl").read_text().splitlines()
            summary = json.loads((Path(tmp) / "t" / "summary.json").read_text())
        self.assertEqual(len(lines), 5)
        self.assertEqual(summary["games"], 5)
        self.assertEqual(summary, result.summary)

    def test_wilson_and_summary(self) -> None:
        lo, hi = wilson(27, 51)
        self.assertAlmostEqual(lo, 0.40, places=2)
        self.assertAlmostEqual(hi, 0.66, places=2)
        s = summarize([
            {"result": "win", "final_score": {"X": 40, "O": 24}, "forfeits": 1},
            {"result": "draw", "final_score": {"X": 32, "O": 32}, "forfeits": 0},
        ])
        self.assertEqual((s["wins"], s["draws"], s["forfeits"], s["mean_margin"]), (1, 1, 1, 8.0))


class CalibrationTests(unittest.TestCase):
    def test_default_configs_include_every_style_once_for_random(self) -> None:
        configs = default_configs()
        self.assertEqual({parse_spec(c).style for c in configs}, set(STYLES))
        self.assertEqual(configs.count("random"), 1)

    def test_fit_ratings_orders_by_results_and_anchors_greedy(self) -> None:
        results = (
            [("strong", "greedy", "win")] * 8 + [("strong", "greedy", "loss")] * 2
            + [("greedy", "weak", "win")] * 8 + [("greedy", "weak", "loss")] * 2
        )
        r = fit_ratings(results)
        self.assertAlmostEqual(r["greedy"], 1000.0, places=6)
        self.assertGreater(r["strong"], r["greedy"])
        self.assertGreater(r["greedy"], r["weak"])


class FakeModel:
    """Returns scripted answers in order; records every prompt it saw."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.prompts: list[list[dict[str, str]]] = []

    def __call__(self, messages, temperature):
        self.prompts.append(messages)
        return (self.answers.pop(0) if self.answers else "nonsense"), {"completion_tokens": 5}


class ModelPlayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.board = new_board()
        self.legal = legal_moves(self.board, NOVA)

    def test_parse_answer(self) -> None:
        self.assertEqual(parse_answer(answer_json("d3"), self.legal), ("d3", ""))
        self.assertEqual(parse_answer('sure: {"tool_name": "play_reversi", "arguments": {"move": "D3"}}', self.legal), ("d3", ""))
        self.assertEqual(parse_answer("d3", self.legal), (None, "no_json"))
        self.assertEqual(parse_answer('{"tool_name": "reflect", "arguments": {}}', self.legal), (None, "wrong_tool"))
        self.assertEqual(parse_answer(answer_json("z9"), self.legal), (None, "not_a_square"))
        self.assertEqual(parse_answer(answer_json("a1"), self.legal), (None, "illegal_move"))
        self.assertEqual(parse_answer("{not json}", self.legal), (None, "bad_json"))

    def test_prompt_shows_board_and_legal_moves(self) -> None:
        messages = build_messages(self.board, self.legal)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("play_reversi", messages[0]["content"])
        for sq in self.legal:
            self.assertIn(sq, messages[1]["content"])
        self.assertIn("*", messages[1]["content"])

    def test_good_answer_is_played(self) -> None:
        fake = FakeModel([answer_json("f5")])
        player = ModelPlayer(fake, name="fake")
        self.assertEqual(player.choose(self.board, NOVA, self.legal, random.Random(0)), "f5")
        self.assertFalse(player.last_detail["forfeit"])
        self.assertEqual(len(player.last_detail["attempts"]), 1)

    def test_retry_then_success_carries_the_correction(self) -> None:
        fake = FakeModel([answer_json("a1"), answer_json("c4")])
        player = ModelPlayer(fake, name="fake")
        self.assertEqual(player.choose(self.board, NOVA, self.legal, random.Random(0)), "c4")
        self.assertEqual([a["error"] for a in player.last_detail["attempts"]], ["illegal_move", ""])
        self.assertIn("illegal_move", fake.prompts[1][-1]["content"])

    def test_forfeit_after_retries_plays_a_random_legal_move(self) -> None:
        fake = FakeModel(["no", "still no", "never"])
        player = ModelPlayer(fake, name="fake", max_retries=2)
        sq = player.choose(self.board, NOVA, self.legal, random.Random(0))
        self.assertIn(sq, self.legal)
        self.assertTrue(player.last_detail["forfeit"])
        self.assertEqual(len(fake.prompts), 3)

    def test_model_player_in_a_full_game_records_forfeits(self) -> None:
        game = play_game(ModelPlayer(FakeModel([]), name="fake"), parse_spec("greedy"), 5)
        x_moves = [m for m in game["moves"] if m["player"] == NOVA and m["square"] != "pass"]
        self.assertTrue(all(m["forfeit"] for m in x_moves))
        self.assertEqual(game["forfeits"], len(x_moves))
        self.assertEqual(x_moves[0]["prompt_version"], "arena-v1")

    def test_model_player_refuses_to_play_o(self) -> None:
        with self.assertRaises(ValueError):
            ModelPlayer(FakeModel([]), name="fake").choose(self.board, OPPONENT, legal_moves(self.board, OPPONENT), random.Random(0))


if __name__ == "__main__":
    unittest.main()
