import numpy as np
from typing import List, Tuple

WIDTH, HEIGHT = 7, 6
SIZE = WIDTH * HEIGHT
# Per Pascal Pons' bitboard formulation
COL_HEIGHT = HEIGHT + 1

COL_MASKS = [((1 << COL_HEIGHT) - 1) << (c * COL_HEIGHT) for c in range(WIDTH)]
BOTTOM_MASKS = [1 << (c * COL_HEIGHT) for c in range(WIDTH)]
TOP_MASKS = [1 << (HEIGHT - 1 + c * COL_HEIGHT) for c in range(WIDTH)]
ALL_MASK = sum(COL_MASKS)

class Connect4:
    """
    Bitboards:
      - mask: all stones
      - current: stones of player to move (i.e., after flipping at play())
    Players: +1 (current), -1 (opponent). We maintain 'player' for perspective.
    """
    __slots__ = ("current", "mask", "player")

    def __init__(self):
        self.current = 0  # bitboard of current player
        self.mask = 0     # bitboard of all stones
        self.player = 1   # +1 (to move) or -1

    def clone(self) -> "Connect4":
        g = Connect4()
        g.current, g.mask, g.player = self.current, self.mask, self.player
        return g

    def key(self):
        return (self.current, self.mask)

    def legal_moves(self) -> List[int]:
        return [c for c in range(WIDTH) if (self.mask & TOP_MASKS[c]) == 0]

    def can_play(self, col: int) -> bool:
        return (self.mask & TOP_MASKS[col]) == 0

    def play(self, col: int):
        """Apply a move for the current player at column col."""
        assert 0 <= col < WIDTH and self.can_play(col)
        # bit of the lowest empty cell in this column
        move = (self.mask + BOTTOM_MASKS[col]) & COL_MASKS[col]
        # next turn: opponent becomes current (xor with mask), then add move to mask
        self.current ^= self.mask
        self.mask |= move
        self.player *= -1

    def last_player_bb(self) -> int:
        """Bitboard of the player who just moved (previous player)."""
        return self.current ^ self.mask

    @staticmethod
    def _is_win(bb: int) -> bool:
        """Check 4 in a row on bitboard bb."""
        # horizontal
        m = bb & (bb >> COL_HEIGHT)
        if m & (m >> (2 * COL_HEIGHT)):
            return True
        # diagonal /
        m = bb & (bb >> (COL_HEIGHT - 1))
        if m & (m >> (2 * (COL_HEIGHT - 1))):
            return True
        # diagonal \
        m = bb & (bb >> (COL_HEIGHT + 1))
        if m & (m >> (2 * (COL_HEIGHT + 1))):
            return True
        # vertical
        m = bb & (bb >> 1)
        if m & (m >> 2):
            return True
        return False

    def terminal(self) -> Tuple[bool, int]:
        """
        Returns (is_terminal, winner)
        winner: +1 if current-to-move would see previous player lost, -1 if previous player won, 0 draw, None if not terminal
        We define from perspective before flip: after play(), player flipped already.
        """
        # previous player stones:
        prev = self.last_player_bb()
        if self._is_win(prev):
            # previous player just created 4-in-a-row; they are -self.player (since we flipped)
            return True, -self.player
        if self.mask == ALL_MASK:
            return True, 0
        return False, None

    # ---------- Helpers for NN / visual ----------
    def to_planes(self) -> np.ndarray:
        """Return 2x6x7 planes: [current, opponent]."""
        cur = self.current
        opp = self.mask ^ cur
        planes = np.zeros((2, HEIGHT, WIDTH), dtype=np.float32)
        for c in range(WIDTH):
            col_all = (self.mask >> (c * COL_HEIGHT)) & ((1 << COL_HEIGHT) - 1)
            col_cur = (cur >> (c * COL_HEIGHT)) & ((1 << COL_HEIGHT) - 1)
            for r in range(HEIGHT):
                bit = 1 << r
                if col_all & bit:
                    if col_cur & bit:
                        planes[0, r, c] = 1.0
                    else:
                        planes[1, r, c] = 1.0
        return planes

    def pretty(self) -> str:
        rows = []
        cur = self.current
        opp = self.mask ^ cur
        for r in reversed(range(HEIGHT)):
            row = []
            for c in range(WIDTH):
                bit = 1 << (r + c * COL_HEIGHT)
                if self.mask & bit:
                    row.append('X' if (cur & bit) else 'O')
                else:
                    row.append('.')
            rows.append(' '.join(row))
        return '\n'.join(rows)
