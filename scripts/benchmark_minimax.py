from c4.env import Connect4
from c4.minimax import best_move
import time

def main():
    g = Connect4()
    n = 20
    st = time.time()
    for i in range(n):
        a,_ = best_move(g, depth=7)
        g.play(a)
        t,w = g.terminal()
        if t: break
    print(f"{i+1} plies in {time.time()-st:.2f}s")

if __name__ == "__main__":
    main()
