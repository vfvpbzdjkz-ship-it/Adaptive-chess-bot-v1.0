# OUROBOROS — Self-Learning Lichess Chess Bot

A chess bot that plays on Lichess and **learns from every game it plays**. No hand-crafted
strategy, no external engines, no databases. Just AlphaZero-style self-play reinforcement
learning + real Lichess games + opponent-specific adaptation.

---

## Honest Expectations

**Please read this before you get excited.**

From-scratch self-play RL on consumer hardware is slow. Here is the realistic trajectory:

| Timeline | What to expect |
|---|---|
| Day 1–3 | Stops hanging pieces randomly |
| Week 1–3 | Coherent, club-ish play emerges |
| Months+ | Keeps improving slowly, diminishing returns |

**It will NOT beat Stockfish through general strength.** That requires vastly more compute
than a home machine provides in any reasonable timeframe.

**Its real superpower**: opponent-adaptive play. Given enough games against the same
deterministic bot, Ouroboros learns exactly what lines beat it — exploiting weaknesses far
above its general Elo level.

The internal Elo shown in the status line is **self-relative** (based on wins between
checkpoints). It is NOT a real Lichess rating. Do not use it to compare against other players.

---

## Setup

### Requirements

- Python 3.11 or newer
- Internet connection (for Lichess)
- A **brand new, empty Lichess account** designated as your bot account

### One-Command Start

**macOS / Linux:**
```bash
./run.sh
```

**Windows:**
```
run.bat
```

On first launch, an interactive wizard will:
1. Ask for your Lichess API token
2. Upgrade your account to BOT status (permanent — use a fresh account!)
3. Detect your hardware and choose a network size
4. Run a seed bootstrap (~10–30 min on CPU, much less on GPU)
5. Start playing

Second and subsequent launches skip straight to playing.

### Getting Your Lichess API Token

1. Create a **new, empty** Lichess account (not your main account)
2. Go to: `https://lichess.org/account/oauth/token/create`
3. Enable scopes: **bot:play**, **challenge:read**, **challenge:write**
4. Copy the token and paste it into the wizard

---

## Run Modes

| Mode | Command | Description |
|---|---|---|
| `auto` (default) | `./run.sh` | Plays Lichess games + trains in background. Leave running for weeks. |
| `play` | `./run.sh play` | Lichess only, no background training. For weak hardware. |
| `train` | `./run.sh train` | Pure offline self-play + training. No internet needed. |

---

## How It Learns

1. **Self-play**: Worker processes play games against themselves using MCTS guided by a
   residual neural network. Training data accumulates in a disk-backed ring buffer.

2. **Real games**: Every Lichess game adds positions to the training buffer at 4× weight
   (real opponents matter more than self-play). If we lose, the opponent's moves are added
   as imitation data.

3. **Opponent profiles**: Every opponent gets a persistent profile in SQLite tracking their
   opening habits, our win rate after each of our moves against them, and (for bots)
   determinism score. This steers future games.

4. **Anti-bot exploitation**: For deterministic bots, Ouroboros memorizes winning lines,
   detects deviations, and systematically explores branches after losses until it finds a
   winning path.

5. **Checkpointing**: Every 1000 training steps → checkpoint saved. Every 5000 steps →
   ladder match (40 games, latest vs best). If latest wins 55%+, it becomes the new best.

---

## Configuration

The **only** file you may need to edit is `data/config.json`. You never need to touch any
Python source files.

To re-run the setup wizard: delete `data/config.json` and run `./run.sh`.

### Key Config Options

```json
{
  "mode": "auto",
  "hardware_profile": "small",
  "matchmaker_enabled": true,
  "accept_rated": true,
  "accept_casual": true,
  "chat_enabled": true,
  "winner_imitation": true,
  "max_concurrent_games": 1
}
```

---

## Architecture

- **Search**: MCTS with PUCT selection, Dirichlet noise during self-play
- **Network**: Residual CNN (AlphaZero-style) with policy + value heads
- **No minimax, no alpha-beta, no external engine, ever**
- **Storage**: SQLite for profiles/games, numpy memmap for replay buffer, PyTorch `.pt` for weights
- **Crash-safe**: atomic writes + WAL SQLite; resume cleanly from any kill signal

---

## Known Limitations

- Weak at first — this is by design and expected
- Internal Elo is not a real Lichess rating
- On CPU, each self-play game takes longer, slowing learning
- The first week is the hardest to watch; patience is the main requirement

---

## FAQ

**Q: Can I run this on a cloud server?**
A: Yes. `./run.sh train` works without internet. Use `auto` once you add the token.

**Q: What if I kill the process mid-game?**
A: On reconnect, the event loop re-attaches to ongoing games automatically.

**Q: Can it play chess.com?**
A: No. Bots violate chess.com's Terms of Service. Lichess only.

**Q: How do I reset everything?**
A: Delete the `data/` directory. This loses all learned weights and profiles.

**Q: The bot plays random moves!**
A: Normal for the first hours. Let it run; the seed bootstrap should give a reasonable start.

---

## File Layout

```
ouroboros/
├── run.sh / run.bat       # One-command bootstrap
├── main.py                # Entry point
├── requirements.txt       # Pinned dependencies
├── ouroboros/
│   ├── engine/            # MCTS, network, encoding, time management
│   ├── learning/          # Buffer, self-play, trainer, online learning, seed
│   ├── lichess/           # API client, event loop, game runner, matchmaker
│   └── opponents/         # Profiles, adaptation, anti-bot exploitation
├── tests/                 # Unit + integration tests
└── data/                  # Created at runtime (gitignored)
    ├── config.json
    ├── ouroboros.db
    ├── models/
    ├── buffer/
    └── logs/
```
