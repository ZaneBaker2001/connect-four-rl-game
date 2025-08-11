import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional
from .env import Connect4, WIDTH

@dataclass
class ChildStats:
    N: int = 0
    W: float = 0.0  # total value
    Q: float = 0.0  # mean value
    P: float = 0.0  # prior

@dataclass
class Node:
    prior: float = 1.0
    children: Dict[int, "Node"] = field(default_factory=dict)
    stats: Dict[int, ChildStats] = field(default_factory=dict)
    terminal: bool = False
    value_from_parent: float = 0.0
    expanded: bool = False

class MCTS:
    def __init__(self, net, cpuct=1.25, dirichlet=(0.25, 0.03), device="cpu"):
        self.net = net
        self.cpuct = cpuct
        self.dirichlet = dirichlet
        self.device = device
        self._cache = {}

    def _encode(self, g: Connect4) -> torch.Tensor:
        return torch.from_numpy(g.to_planes()).unsqueeze(0).to(self.device)

    def _root(self, g: Connect4) -> Node:
        k = g.key()
        if k not in self._cache:
            self._cache[k] = Node()
        return self._cache[k]

    @torch.no_grad()
    def _evaluate(self, g: Connect4):
        planes = self._encode(g)  # [1,2,6,7]
        logits, v = self.net(planes)
        pi = torch.softmax(logits[0], dim=-1).cpu().numpy()
        return pi, float(v.item())

    def run(self, game: Connect4, sims=128, temperature=1.0, add_noise=True):
        root_g = game.clone()
        root = self._root(root_g)

        # If not expanded yet, evaluate once
        if not root.expanded:
            term, winner = root_g.terminal()
            root.terminal = term
            if term:
                root.value_from_parent = 1.0 if winner == 1 else (-1.0 if winner == -1 else 0.0)
                root.expanded = True
            else:
                priors, _ = self._evaluate(root_g)
                root.stats = {a: ChildStats(P=priors[a]) for a in range(WIDTH)}
                root.expanded = True

        # Dirichlet noise at the root
        if add_noise:
            eps, alpha = self.dirichlet
            noise = np.random.dirichlet([alpha] * WIDTH)
            for a in range(WIDTH):
                if a in root.stats:
                    root.stats[a].P = (1 - eps) * root.stats[a].P + eps * noise[a]

        for _ in range(sims):
            g = root_g.clone()
            node = root
            path = []

            # Selection
            while node.children:
                # choose action that maximizes Q + U
                N_sum = sum(cs.N for cs in node.stats.values())
                best_a, best_score = None, -1e9
                for a, cs in node.stats.items():
                    if not g.can_play(a):
                        continue
                    U = self.cpuct * cs.P * math.sqrt(N_sum + 1e-8) / (1 + cs.N)
                    score = cs.Q + U
                    if score > best_score:
                        best_score = score
                        best_a = a
                if best_a is None:
                    break
                path.append((node, best_a))
                g.play(best_a)
                node = node.children.get(best_a, Node())

            # Evaluate leaf
            term, winner = g.terminal()
            if term:
                v = 1.0 if winner == 1 else (-1.0 if winner == -1 else 0.0)
                leaf_priors = None
            else:
                leaf_priors, v = self._evaluate(g)

            # Backup (flip value along the path)
            for parent, a in reversed(path):
                cs = parent.stats[a]
                cs.N += 1
                cs.W += v
                cs.Q = cs.W / cs.N
                v = -v  # opponent's perspective

            # Expand leaf into the tree
            if not term:
                # attach new node at last edge
                if path:
                    parent, a = path[-1]
                    parent.children[a] = Node()
                    parent = parent.children[a]
                else:
                    parent = node
                parent.stats = {a: ChildStats(P=leaf_priors[a]) for a in range(WIDTH)}
                parent.expanded = True

        # Build policy from visit counts at root, masked to legal
        counts = np.zeros(WIDTH, dtype=np.float32)
        for a, cs in root.stats.items():
            if game.can_play(a):
                counts[a] = cs.N

        if temperature <= 1e-6:
            a = int(np.argmax(counts))
            probs = np.zeros(WIDTH, dtype=np.float32)
            probs[a] = 1.0
        else:
            p = counts ** (1.0 / temperature)
            probs = p / (p.sum() + 1e-8)
        return probs
