"""Tests for Phase 22 Stage 22.13 — reversi as exogenous input on the tick surface."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from nova.agent.motive import default_motive_state
from nova.agent.reversi import (
    NOVA,
    OPPONENT,
    ReversiController,
    ReversiStore,
    apply_move,
    board_from_rows,
    choose_opponent_move,
    flips_for,
    legal_moves,
    new_board,
    parse_square,
    render_board,
    render_play_result,
    score,
)
from nova.agent.self_state_tick import SelfStateTickEngine
from nova.agent.self_state_tools import (
    CARRYOVER_TOOL_NAMES,
    READ_TOOL_NAMES,
    SELF_STATE_TOOL_NAMES,
    SelfStateToolDispatcher,
    render_read_tool_result,
)
from nova.agent.tool_registry import default_tool_registry
from nova.agent.tools import ToolRequest
from nova.config import GameConfig, NovaConfig
from nova.persona.defaults import default_persona_state, default_self_state


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


class ReversiRulesTests(unittest.TestCase):
    def test_opening_position_and_legal_moves(self) -> None:
        board = new_board()
        self.assertEqual(score(board), {NOVA: 2, OPPONENT: 2})
        self.assertEqual(legal_moves(board, NOVA), ["d3", "c4", "f5", "e6"])
        self.assertEqual(legal_moves(board, OPPONENT), ["e3", "f4", "c5", "d6"])

    def test_parse_square(self) -> None:
        self.assertEqual(parse_square("d3"), (2, 3))
        self.assertEqual(parse_square(" D3 "), (2, 3))
        self.assertEqual(parse_square("h8"), (7, 7))
        self.assertIsNone(parse_square("i1"))
        self.assertIsNone(parse_square("d9"))
        self.assertIsNone(parse_square("3d"))
        self.assertIsNone(parse_square(""))

    def test_apply_move_flips_and_rejects_illegal(self) -> None:
        board = new_board()
        self.assertEqual(apply_move(board, NOVA, 2, 3), 1)  # d3 flips d4
        self.assertEqual(board[3][3], NOVA)
        self.assertEqual(score(board), {NOVA: 4, OPPONENT: 1})
        with self.assertRaises(ValueError):
            apply_move(board, NOVA, 2, 3)  # occupied
        with self.assertRaises(ValueError):
            apply_move(board, NOVA, 0, 0)  # no capture line

    def test_multi_direction_capture(self) -> None:
        rows = [
            "........",
            "........",
            "..OOO...",
            "..OXO...",
            "..OOO...",
            "........",
            "........",
            "........",
        ]
        board = board_from_rows(rows)
        # X at d4 surrounded; X playing b2 captures c3 (diagonal) only if d4 is own: yes.
        self.assertEqual(sorted(flips_for(board, NOVA, 1, 1)), [(2, 2)])
        # X playing d2 captures d3 via d4.
        self.assertEqual(flips_for(board, NOVA, 1, 3), [(2, 3)])

    def test_render_board_marks_legal_squares(self) -> None:
        text = render_board(new_board(), legal=legal_moves(new_board(), NOVA))
        self.assertIn("    a b c d e f g h", text)
        self.assertEqual(text.count("*"), 4)
        self.assertIn("4   . . * O X . . .", text)

    def test_opponent_policies_are_legal_and_seed_deterministic(self) -> None:
        board = new_board()
        for policy in ("random", "greedy"):
            a = choose_opponent_move(board, policy=policy, rng=random.Random(7))
            b = choose_opponent_move(board, policy=policy, rng=random.Random(7))
            self.assertEqual(a, b)
            self.assertIn(a, legal_moves(board, OPPONENT))
        with self.assertRaises(ValueError):
            choose_opponent_move(board, policy="oracle", rng=random.Random(1))

    def test_random_legal_playout_always_terminates_with_full_or_dead_board(self) -> None:
        rng = random.Random(3)
        for _ in range(20):
            board = new_board()
            player = NOVA
            passes = 0
            for _turn in range(200):
                legal = legal_moves(board, player)
                if not legal:
                    passes += 1
                    if passes == 2:
                        break
                    player = OPPONENT if player == NOVA else NOVA
                    continue
                passes = 0
                row, col = parse_square(rng.choice(legal))  # type: ignore[misc]
                apply_move(board, player, row, col)
                player = OPPONENT if player == NOVA else NOVA
            self.assertFalse(legal_moves(board, NOVA))
            self.assertFalse(legal_moves(board, OPPONENT))
            total = score(board)
            self.assertLessEqual(total[NOVA] + total[OPPONENT], 64)


# ---------------------------------------------------------------------------
# Controller + store
# ---------------------------------------------------------------------------


class ReversiControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.controller = ReversiController(
            ReversiStore(self.dir), opponent_policy="greedy", seed=11
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_new_starts_a_game_with_nova_to_move(self) -> None:
        reply = self.controller.play(move="new", session_id="s1")
        self.assertTrue(reply["ok"])
        self.assertTrue(reply["started"])
        self.assertEqual(reply["status"], "active")
        self.assertEqual(reply["legal_moves"], ["d3", "c4", "f5", "e6"])
        self.assertEqual(reply["game_number"], 1)
        self.assertIsNotNone(self.controller.current())

    def test_a_square_with_no_game_starts_and_plays(self) -> None:
        reply = self.controller.play(move="d3", session_id="s1", tick_ref="self_state_tick:t1")
        self.assertTrue(reply["ok"])
        self.assertTrue(reply["started"])
        self.assertEqual(reply["your_move"], "d3")
        self.assertEqual(reply["flipped"], 1)
        self.assertEqual(len(reply["opponent"]), 1)
        self.assertIn(reply["opponent"][0]["square"], ("c3", "e3", "c5"))
        game = self.controller.current()
        assert game is not None
        self.assertEqual(game.moves[0]["player"], NOVA)
        self.assertEqual(game.moves[0]["tick_ref"], "self_state_tick:t1")
        self.assertEqual(game.moves[1]["player"], OPPONENT)
        self.assertTrue(reply["legal_moves"])  # Nova always has a move when shown the board

    def test_state_persists_across_controller_instances(self) -> None:
        self.controller.play(move="d3", session_id="s1")
        again = ReversiController(ReversiStore(self.dir), opponent_policy="greedy", seed=11)
        game = again.current()
        assert game is not None
        self.assertEqual(len(game.moves), 2)
        self.assertEqual(game.seed, 11)

    def test_illegal_and_malformed_moves_are_counted_not_applied(self) -> None:
        self.controller.play(move="new", session_id="s1")
        bad = self.controller.play(move="a1", session_id="s1")
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["error"], "illegal_move")
        junk = self.controller.play(move="zebra", session_id="s1")
        self.assertEqual(junk["error"], "not_a_square")
        game = self.controller.current()
        assert game is not None
        self.assertEqual(game.illegal_attempts, 2)
        self.assertEqual(game.moves, [])
        self.assertEqual(self.controller.record()["illegal_attempts"], 2)

    def test_new_during_a_game_is_refused(self) -> None:
        self.controller.play(move="d3", session_id="s1")
        reply = self.controller.play(move="new", session_id="s1")
        self.assertFalse(reply["ok"])
        self.assertEqual(reply["error"], "game_in_progress")

    def test_resign_finishes_as_loss_and_clears_current(self) -> None:
        self.controller.play(move="d3", session_id="s1")
        reply = self.controller.play(move="resign", session_id="s1")
        self.assertTrue(reply["ok"])
        self.assertEqual(reply["status"], "finished")
        self.assertEqual(reply["result"], "loss")
        self.assertIsNone(self.controller.current())
        rec = self.controller.record()
        self.assertEqual((rec["games_finished"], rec["losses"]), (1, 1))
        no_game = self.controller.play(move="resign", session_id="s1")
        self.assertEqual(no_game["error"], "no_game")

    def test_full_game_reaches_a_result_and_seed_replays(self) -> None:
        def playout(seed: int) -> list[str]:
            with tempfile.TemporaryDirectory() as tmp:
                ctl = ReversiController(ReversiStore(tmp), opponent_policy="greedy", seed=seed)
                rng = random.Random(seed)
                reply = ctl.play(move="new", session_id="s")
                squares: list[str] = []
                for _ in range(80):
                    if reply["status"] == "finished":
                        break
                    mv = rng.choice(reply["legal_moves"])
                    squares.append(mv)
                    reply = ctl.play(move=mv, session_id="s")
                    self.assertTrue(reply["ok"], reply)
                self.assertEqual(reply["status"], "finished")
                self.assertIn(reply["result"], ("win", "loss", "draw"))
                self.assertEqual(ctl.record()["games_finished"], 1)
                self.assertIsNone(ctl.current())
                finished = ReversiStore(tmp).list_finished()[0]
                self.assertEqual(finished.final_score, reply["score"])
                return squares + [reply["result"]]

        self.assertEqual(playout(5), playout(5))

    def test_prompt_block_with_and_without_a_game(self) -> None:
        empty = self.controller.prompt_block()
        self.assertTrue(empty.startswith("[Reversi]"))
        self.assertIn("No game in progress", empty)
        self.assertIn("'new'", empty)
        self.controller.play(move="d3", session_id="s1")
        block = self.controller.prompt_block()
        self.assertIn("Game 1: you are X", block)
        self.assertIn("Legal moves:", block)
        self.assertIn("Last: you d3 (flipped 1); opponent", block)
        self.assertIn("    a b c d e f g h", block)

    def test_render_play_result_is_compact(self) -> None:
        reply = self.controller.play(move="d3", session_id="s1")
        text = render_play_result(reply)
        self.assertTrue(text.startswith("play_reversi:"))
        self.assertIn("You played d3", text)
        self.assertIn("score: X", text)
        self.assertIn("your legal moves now:", text)
        self.assertLess(len(text), 400)


# ---------------------------------------------------------------------------
# Wiring: dispatcher, registry, carryover, prompt, config
# ---------------------------------------------------------------------------


def _dispatcher(controller=None) -> SelfStateToolDispatcher:
    persona = default_persona_state()
    return SelfStateToolDispatcher(
        self_state=default_self_state(persona),
        motive_state=default_motive_state(session_id="t"),
        soul_block="[Soul]",
        session_id="t",
        reversi_controller=controller,
    )


class ReversiWiringTests(unittest.TestCase):
    def test_tool_name_registered_and_parseable(self) -> None:
        self.assertIn("play_reversi", SELF_STATE_TOOL_NAMES)
        names = {spec.name for spec in default_tool_registry().list_specs()}
        self.assertIn("play_reversi", names)
        req = SelfStateTickEngine().parse(
            raw_text='{"tool_name": "play_reversi", "arguments": {"move": "d3"}}',
            session_id="s",
            tick_id="s:1",
        )
        assert req is not None
        self.assertEqual(req.arguments["move"], "d3")

    def test_dispatch_without_controller_reports_unavailable(self) -> None:
        result = _dispatcher().dispatch(ToolRequest(tool_name="play_reversi", arguments={"move": "d3"}))
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "reversi_unavailable")

    def test_dispatch_with_controller_plays_and_carries_over(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctl = ReversiController(ReversiStore(tmp), seed=1)
            result = _dispatcher(ctl).dispatch(
                ToolRequest(tool_name="play_reversi", arguments={"move": "d3", "comment": "opening"},
                            reason="self_state_tick:s:1")
            )
            self.assertTrue(result["ok"])
            game = ctl.current()
            assert game is not None
            self.assertEqual(game.moves[0]["comment"], "opening")
            self.assertEqual(game.session_id, "t")
            self.assertIn("play_reversi", CARRYOVER_TOOL_NAMES)
            self.assertNotIn("play_reversi", READ_TOOL_NAMES)
            self.assertIn("You played d3", render_read_tool_result("play_reversi", result))

    def _messages(self, **kwargs):
        return SelfStateTickEngine().build_messages(
            session_id="s",
            tick_id="s:1",
            trigger="daemon_tick",
            self_context_block="[Self-Context]\nfocus: x",
            recent_heartbeats=[],
            **kwargs,
        )

    def test_prompt_is_byte_identical_when_disabled(self) -> None:
        for register in ("assertion", "exploratory"):
            default = self._messages(register=register)
            explicit = self._messages(register=register, reversi_enabled=False, reversi_block="[Reversi]\nignored")
            self.assertEqual(default, explicit)
            system = default[0]["content"]
            self.assertNotIn("reversi", system.lower())
            self.assertNotIn("{game_tool}", system)
            self.assertNotIn("[Reversi]", default[1]["content"])

    def test_prompt_carries_menu_tool_text_and_block_when_enabled(self) -> None:
        for register in ("assertion", "exploratory"):
            msgs = self._messages(register=register, reversi_enabled=True, reversi_block="[Reversi]\nboard")
            system, user = msgs[0]["content"], msgs[1]["content"]
            self.assertIn(", play_reversi.", system)
            self.assertIn("- play_reversi (arguments: 'move'", system)
            self.assertNotIn("{game_tool}", system)
            self.assertIn("[Reversi]\nboard", user)

    def test_config_defaults_off_and_validates_policy(self) -> None:
        cfg = NovaConfig()
        self.assertFalse(cfg.game.reversi_enabled)
        self.assertEqual(cfg.game.reversi_opponent, "greedy")
        cfg.model.model_path = "x"
        cfg.validate()
        cfg.game = GameConfig(reversi_enabled=True, reversi_opponent="oracle")
        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# End to end: one daemon tick with the game enabled
# ---------------------------------------------------------------------------

from nova.types import GenerationRequest, GenerationResult  # noqa: E402
from tests.test_runtime_smoke import FakeBackend, build_test_runtime  # noqa: E402


class ReversiMoveBackend(FakeBackend):
    """Answers every tick with a play_reversi call and keeps the prompt."""

    def __init__(self, move: str = "d3") -> None:
        super().__init__()
        self.move = move
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.generate_calls += 1
        self.requests.append(request)
        return GenerationResult(
            model_id=request.model_id,
            raw_text='{"tool_name": "play_reversi", "arguments": {"move": "%s"}}' % self.move,
            finish_reason="stop",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=12,
            latency_ms=1,
            metadata={"backend": "fake"},
        )


class ReversiTickIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.data_dir = base / "data"
        self.backend = ReversiMoveBackend()
        self.runtime = build_test_runtime(
            data_dir=self.data_dir, log_dir=base / "logs", backend=self.backend
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @staticmethod
    def _prompt_text(request: GenerationRequest) -> str:
        messages = getattr(request, "messages", None) or []
        return "\n".join(str(m.get("content", "")) for m in messages) or request.prompt

    def test_disabled_runtime_has_no_board_and_answers_unavailable(self) -> None:
        self.assertIsNone(self.runtime.reversi_controller)
        self.runtime.start(session_id="reversi-off")
        self.runtime.start_operational_autonomy(max_ticks=0)
        tick = self.runtime.model_self_state_tick()
        self.runtime.close()
        self.assertNotIn("[Reversi]", self._prompt_text(self.backend.requests[-1]))
        self.assertEqual(tick.adapter_audit["tool_requested"], "play_reversi")
        self.assertEqual(tick.adapter_audit["tool_result"]["error"], "reversi_unavailable")
        self.assertFalse((self.data_dir / "games").exists())

    def test_enabled_runtime_shows_board_plays_and_carries_over(self) -> None:
        # The live path constructs this in __init__ from config.game; the
        # test helper builds its config without that section, so wire the
        # controller the same way the runtime would.
        self.runtime.reversi_controller = ReversiController(
            ReversiStore(self.data_dir / "games"), opponent_policy="greedy", seed=3
        )
        self.runtime.start(session_id="reversi-on")
        self.runtime.start_operational_autonomy(max_ticks=0)

        first = self.runtime.model_self_state_tick()
        prompt_1 = self._prompt_text(self.backend.requests[-1])
        self.assertIn(", play_reversi.", prompt_1)
        self.assertIn("[Reversi]\nNo game in progress.", prompt_1)
        self.assertTrue(first.adapter_audit["tool_executed"])
        self.assertTrue(first.adapter_audit["tool_result"]["ok"])
        self.assertEqual(first.adapter_audit["tool_result"]["your_move"], "d3")

        self.backend.move = "a1"  # illegal now: exercises the error path on a live board
        second = self.runtime.model_self_state_tick()
        prompt_2 = self._prompt_text(self.backend.requests[-1])
        self.runtime.close()
        self.assertIn("[Reversi]\nGame 1: you are X", prompt_2)
        self.assertIn("Legal moves:", prompt_2)
        self.assertIn("[Results of your recent tool calls]", prompt_2)
        self.assertIn("play_reversi:\n  You played d3 (flipped 1).", prompt_2)
        self.assertEqual(second.adapter_audit["tool_result"]["error"], "illegal_move")

        store = ReversiStore(self.data_dir / "games")
        game = store.load_current()
        assert game is not None
        self.assertEqual(game.session_id, "reversi-on")
        self.assertEqual(game.illegal_attempts, 1)
        self.assertEqual([m["player"] for m in game.moves], [NOVA, OPPONENT])
        self.assertTrue(game.moves[0]["tick_ref"].startswith("self_state_tick:reversi-on:"))
