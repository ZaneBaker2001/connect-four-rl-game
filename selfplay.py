import numpy as np
from typing import List, Tuple
from .env import Connect4
from .mcts import MCTS

def play_episode(mcts: MCTS, sims_train=96, temp_moves=10, seed=None) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    rng = np.random.default_rng(seed)
    g = Connect4()
    history = []  # [(planes, pi, player), ...]
    move_idx = 0

    while True:
        temperature = 1.0 if move_idx < temp_moves else 0.25
        pi = mcts.run(g, sims=sims_train, temperature=temperature, add_noise=True)
        # sample an action from pi
        legal = g.legal_moves()
        probs = pi.copy()
        probs[[a for a in range(len(probs)) if a not in legal]] = 0.0
        probs = probs / (probs.sum() + 1e-8)
        a = int(rng.choice(len(probs), p=probs))
        planes = g.to_planes()
        history.append((planes, pi, g.player))
        g.play(a)
        move_idx += 1

        term, winner = g.terminal()
        if term:
            z_final = float(winner)  # +1, -1, or 0
            # assign outcomes from each state perspective
            out = []
            for planes, pi_h, player in history:
                out.append((planes, pi_h, z_final * player))
            return out
