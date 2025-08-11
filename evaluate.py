import numpy as np
from tqdm import trange
import torch
from .env import Connect4
from .mcts import MCTS
from .net import PolicyValueNet
from .minimax import best_move as minimax_best

def load_model(path, device=None, blocks=6, ch=64):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyValueNet(blocks=blocks, ch=ch).to(device)
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model, device

def play_game(agent_policy, opponent_policy):
    g = Connect4()
    while True:
        a = agent_policy(g) if g.player == 1 else opponent_policy(g)
        g.play(a)
        term, winner = g.terminal()
        if term:
            return winner

def agent_mcts_policy(model, device, sims_eval=256, temperature=0.0):
    mcts = MCTS(model, device=device)
    def policy(g: Connect4):
        pi = mcts.run(g, sims=sims_eval, temperature=temperature, add_noise=False)
        legal = g.legal_moves()
        pi[[a for a in range(len(pi)) if a not in legal]] = 0.0
        a = int(np.argmax(pi))
        return a
    return policy

def minimax_policy(depth=7):
    def pol(g: Connect4):
        a, _ = minimax_best(g, depth=depth)
        return a
    return pol

def arena_vs_minimax(ckpt_path, games=200, sims_eval=256, minimax_depth=7, blocks=6, ch=64):
    model, device = load_model(ckpt_path, blocks=blocks, ch=ch)
    agent_pol = agent_mcts_policy(model, device, sims_eval=sims_eval)

    wins = draws = losses = 0
    for i in trange(games, desc="Arena"):
        if i % 2 == 0:
            w = play_game(agent_pol, minimax_policy(minimax_depth))
            if w == 1: wins += 1
            elif w == 0: draws += 1
            else: losses += 1
        else:
            w = play_game(minimax_policy(minimax_depth), agent_pol)
            if w == -1: wins += 1   # agent was second; winner -1 means agent
            elif w == 0: draws += 1
            else: losses += 1
    return wins, draws, losses
