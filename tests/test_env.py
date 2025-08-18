import pytest
from src.env import Connect4

def test_empty_legal():
    g = Connect4()
    assert set(g.legal_moves()) == set(range(7))

def test_play_and_terminal():
    g = Connect4()
    # vertical win in col 3 for first player
    for _ in range(3):
        g.play(3); g.play(4)
    g.play(3)
    term, winner = g.terminal()
    assert term and winner == -1  # previous player (first) won; current is -1
