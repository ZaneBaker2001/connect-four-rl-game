# Simple alpha-beta minimax baseline with shallow eval.
from functools import lru_cache
from .env import Connect4, WIDTH, HEIGHT, COL_MASKS, TOP_MASKS, BOTTOM_MASKS, COL_HEIGHT

def _winning(bb: int) -> bool:
    # same patterns as env
    m = bb & (bb >> COL_HEIGHT)
    if m & (m >> (2 * COL_HEIGHT)): return True
    m = bb & (bb >> (COL_HEIGHT - 1))
    if m & (m >> (2 * (COL_HEIGHT - 1))): return True
    m = bb & (bb >> (COL_HEIGHT + 1))
    if m & (m >> (2 * (COL_HEIGHT + 1))): return True
    m = bb & (bb >> 1)
    if m & (m >> 2): return True
    return False

def heuristic(g: Connect4) -> int:
    """Very simple: center control + mobility."""
    score = 0
    # center column bonus
    center_c = WIDTH // 2
    cur = g.current
    opp = g.mask ^ cur
    for r in range(HEIGHT):
        bit = 1 << (r + center_c * COL_HEIGHT)
        if cur & bit: score += 2
        if opp & bit: score -= 2
    # mobility
    score += len(g.legal_moves()) - 3
    return score

def best_move(g: Connect4, depth=6):
    """Return (best_action, value) for current player."""
    def negamax(state: Connect4, d: int, alpha: int, beta: int) -> int:
        term, winner = state.terminal()
        if term:
            if winner == 1: return 10_000 - (6*7 - popcount(state.mask))
            if winner == -1: return -10_000 + (6*7 - popcount(state.mask))
            return 0
        if d == 0: return heuristic(state)

        best = -1_000_000
        for a in order_moves(state):
            s2 = state.clone()
            s2.play(a)
            val = -negamax(s2, d - 1, -beta, -alpha)
            if val > best: best = val
            if best > alpha: alpha = best
            if alpha >= beta: break
        return best

    def order_moves(state: Connect4):
        # prefer center, then adjacent
        center = WIDTH // 2
        order = [center]
        for i in range(1, center + 1):
            if center - i >= 0: order.append(center - i)
            if center + i < WIDTH: order.append(center + i)
        return [a for a in order if state.can_play(a)]

    def popcount(x):
        return x.bit_count()

    best_a, best_v = None, -1_000_000
    for a in order_moves(g):
        s2 = g.clone(); s2.play(a)
        v = -negamax(s2, depth - 1, -1_000_000, 1_000_000)
        if v > best_v:
            best_v, best_a = v, a
    return best_a, best_v
