# AlphaZero‑style Connect Four (Self‑Play + MCTS + PyTorch)

Trains a Connect Four agent from scratch using **self‑play** and **Monte Carlo Tree Search (MCTS)** with a compact **policy/value ResNet**. Runs on CPU (slow) or GPU (recommended). Includes evaluation vs a minimax baseline, tests, and a simple CLI.



##  Highlights
- End‑to‑end **AlphaZero-style** loop: self‑play → training → evaluation.
- **ResNet‑lite** policy/value network (PyTorch).
- **PUCT MCTS** with Dirichlet root noise, temperature control.
- **Minimax** baseline for head‑to‑head evaluation.
- Clean **CLI** (train/eval/play) + **pytest** smoke tests.



##  Requirements
- Python 3.9+
- Recommended: CUDA GPU for training speed

Install deps:
```
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```



## Project Structure
```
connect-four-rl-game/
  README.md
  requirements.txt
  src/
    env.py            # bitboard game logic
    net.py            # policy/value ResNet-lite
    mcts.py           # PUCT Monte Carlo Tree Search
    replay.py         # simple FIFO replay buffer
    selfplay.py       # self-play episode generator
    train.py          # optimization loop
    minimax.py        # alpha-beta baseline
    evaluate.py       # arenas vs minimax
    cli.py            # train / evaluate / play
  tests/
    test_env.py
    test_mcts.py
  scripts/
    benchmark_minimax.py
```



##  Quickstart

### 1) Run tests
```bash
pytest -q
```

### 2) Train (small demo run)
```bash
python -m src.cli train \
  --runs-dir runs/c4 \
  --epochs 5 \
  --selfplay-games 200 \
  --mcts-sims 96 \
  --blocks 6 \
  --channels 64 \
  --batch-size 256
```
- Checkpoints are saved to `--runs-dir` as `best.pt`.

### 3) Evaluate vs minimax
```bash
python -m src.cli evaluate \
  --runs-dir runs/c4 \
  --checkpoint best.pt \
  --games 200 \
  --mcts-sims 256 \
  --minimax-depth 7
```
Outputs a W/D/L summary.

### 4) Play against your agent
```bash
python -m src.cli play \
  --runs-dir runs/c4 \
  --checkpoint best.pt \
  --mcts-sims 256 \
  --human first   # or: --human second
```
Type a column index (0–6) when prompted.



##  How it works (quick)

**State encoding:** 2 planes (current player stones, opponent stones) in a 6×7 grid derived from a fast bitboard representation.

**Network:** Small ResNet (default 6 blocks, 64 channels). Heads:
- **Policy:** logits over 7 columns.
- **Value:** scalar in [-1, 1].

**MCTS (PUCT):**
Score per action `a` from node `s`:  
`Q(s,a) + c_puct * P(s,a) * sqrt(Σ_b N(s,b)) / (1 + N(s,a))`  
Root uses **Dirichlet noise** for exploration during self‑play; **temperature** decays after opening moves.

**Targets from self‑play:**
- Policy target = normalized **visit counts** at the root.
- Value target = final game outcome from the state’s current player perspective.

**Loss:** `KL(policy_logits, π_visit)` + `λ * MSE(value, z)` (λ=0.5 by default).



##  Useful knobs
- **Strength**: increase `--selfplay-games`, `--epochs`, and `--mcts-sims` (train & eval).
- **Model size**: `--blocks`, `--channels`.
- **Evaluation**: raise `--minimax-depth` for a tougher baseline.



