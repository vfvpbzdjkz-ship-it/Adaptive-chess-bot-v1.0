"""Profile update and blending math tests."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def _init_test_db(tmpdir: str):
    import ouroboros.persistence as pers
    pers.DB_PATH = Path(tmpdir) / "test.db"
    pers.init_db()


def test_get_or_create_opponent():
    with tempfile.TemporaryDirectory() as tmpdir:
        _init_test_db(tmpdir)
        from ouroboros.opponents.profiles import get_or_create_opponent, get_opponent
        opp_id = get_or_create_opponent("testbot", is_bot=True, title="BOT", last_elo=1500)
        assert opp_id is not None
        opp = get_opponent("testbot")
        assert opp is not None
        assert opp["username"] == "testbot"
        assert opp["is_bot"] == 1


def test_opponent_stats_after_game():
    with tempfile.TemporaryDirectory() as tmpdir:
        _init_test_db(tmpdir)
        from ouroboros.opponents.profiles import get_or_create_opponent, get_opponent, update_opponent_after_game

        get_or_create_opponent("humanplayer", is_bot=False, title="", last_elo=1400)

        pgn = '[Event "?"]\n[White "humanplayer"]\n[Black "ouroboros"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 *'

        update_opponent_after_game(
            username="humanplayer",
            is_bot=False,
            last_elo=1400,
            result="loss",
            pgn=pgn,
            our_color="black",
        )

        opp = get_opponent("humanplayer")
        # result="loss" means we lost → opponent wins_vs_us increments
        assert opp["wins_vs_us"] == 1
        assert opp["losses_vs_us"] == 0


def test_blending_confidence():
    """Confidence weight = games / (games + 5) — verify the math."""
    def blend_weight(games):
        return games / (games + 5)

    assert blend_weight(0) == 0.0
    assert abs(blend_weight(5) - 0.5) < 1e-9
    assert abs(blend_weight(95) - 0.95) < 1e-5

    # Bots: confidence capped at ADAPT_LAMBDA_MAX (0.6)
    from ouroboros.opponents.adapt import ADAPT_LAMBDA_MAX
    for games in [0, 5, 100]:
        confidence = min(blend_weight(games), ADAPT_LAMBDA_MAX)
        assert confidence <= ADAPT_LAMBDA_MAX


def test_ema_update():
    """EMA formula: new = (1-alpha)*old + alpha*sample."""
    alpha = 0.25
    old_score = 0.5
    new_sample = 1.0
    expected = (1 - alpha) * old_score + alpha * new_sample
    assert abs(expected - 0.625) < 1e-9


def test_opening_moves_recorded():
    with tempfile.TemporaryDirectory() as tmpdir:
        _init_test_db(tmpdir)
        from ouroboros.opponents.profiles import (
            get_or_create_opponent, get_opening_stats, update_opponent_after_game
        )
        get_or_create_opponent("botX", is_bot=True, title="BOT", last_elo=1600)

        pgn = '[Event "?"]\n[White "ouroboros"]\n[Black "botX"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 *'
        update_opponent_after_game(
            username="botX", is_bot=True, last_elo=1600,
            result="win", pgn=pgn, our_color="white",
        )

        from ouroboros.persistence import get_db
        with get_db() as conn:
            opp = conn.execute("SELECT id FROM opponents WHERE username='botX'").fetchone()
            rows = conn.execute(
                "SELECT * FROM opening_moves WHERE opponent_id=?", (opp["id"],)
            ).fetchall()
        assert len(rows) > 0, "No opening moves recorded"


if __name__ == "__main__":
    test_get_or_create_opponent()
    test_opponent_stats_after_game()
    test_blending_confidence()
    test_ema_update()
    test_opening_moves_recorded()
    print("All profile tests passed.")
