import torch
from c4.env import Connect4
from c4.net import PolicyValueNet
from c4.mcts import MCTS

def test_mcts_runs():
    model = PolicyValueNet()
    model.eval()
    mcts = MCTS(model)
    g = Connect4()
    pi = mcts.run(g, sims=8, temperature=1.0)
    assert pi.shape[0] == 7
    assert abs(float(pi.sum()) - 1.0) < 1e-5
