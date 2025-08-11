import os
import argparse
import torch
import numpy as np
from .train import train_loop, save_ckpt, load_ckpt
from .net import PolicyValueNet
from .evaluate import arena_vs_minimax
from .env import Connect4
from .mcts import MCTS

def cmd_train(args):
    best = train_loop(
        runs_dir=args.runs_dir,
        epochs=args.epochs,
        selfplay_games=args.selfplay_games,
        sims=args.mcts_sims,
        blocks=args.blocks,
        ch=args.channels,
        batch_size=args.batch_size
    )
    print("Best checkpoint:", best)

def cmd_evaluate(args):
    ckpt = os.path.join(args.runs_dir, args.checkpoint)
    w,d,l = arena_vs_minimax(
        ckpt_path=ckpt,
        games=args.games,
        sims_eval=args.mcts_sims,
        minimax_depth=args.minimax_depth,
        blocks=args.blocks,
        ch=args.channels
    )
    print(f"vs minimax (depth {args.minimax_depth}): {w}W {d}D {l}L")

def cmd_play(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PolicyValueNet(blocks=args.blocks, ch=args.channels).to(device)
    load_ckpt(model, os.path.join(args.runs_dir, args.checkpoint), map_location=device)
    model.eval()
    mcts = MCTS(model, device=device)

    g = Connect4()
    human_as = -1 if args.human == "second" else 1
    print("You are", "O (second)" if human_as == -1 else "X (first)")
    while True:
        print("\nBoard:\n" + g.pretty())
        if g.player == human_as:
            legal = g.legal_moves()
            move = None
            while move not in legal:
                try:
                    move = int(input(f"Your move {legal}: "))
                except Exception:
                    move = None
        else:
            pi = mcts.run(g, sims=args.mcts_sims, temperature=0.0, add_noise=False)
            move = int(np.argmax(pi))
            print(f"AI plays {move}")
        g.play(move)
        term, winner = g.terminal()
        if term:
            print("\nFinal:\n" + g.pretty())
            if winner == 0: print("Draw.")
            else:
                print("Winner:", "You" if winner == human_as else "AI")
            break

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    tp = sub.add_parser("train")
    tp.add_argument("--runs-dir", default="runs/c4")
    tp.add_argument("--epochs", type=int, default=5)
    tp.add_argument("--selfplay-games", type=int, default=200)
    tp.add_argument("--mcts-sims", type=int, default=96)
    tp.add_argument("--blocks", type=int, default=6)
    tp.add_argument("--channels", type=int, default=64)
    tp.add_argument("--batch-size", type=int, default=256)
    tp.set_defaults(func=cmd_train)

    ep = sub.add_parser("evaluate")
    ep.add_argument("--runs-dir", default="runs/c4")
    ep.add_argument("--checkpoint", default="best.pt")
    ep.add_argument("--games", type=int, default=200)
    ep.add_argument("--mcts-sims", type=int, default=256)
    ep.add_argument("--minimax-depth", type=int, default=7)
    ep.add_argument("--blocks", type=int, default=6)
    ep.add_argument("--channels", type=int, default=64)
    ep.set_defaults(func=cmd_evaluate)

    pp = sub.add_parser("play")
    pp.add_argument("--runs-dir", default="runs/c4")
    pp.add_argument("--checkpoint", default="best.pt")
    pp.add_argument("--mcts-sims", type=int, default=256)
    pp.add_argument("--blocks", type=int, default=6)
    pp.add_argument("--channels", type=int, default=64)
    pp.add_argument("--human", choices=["first","second"], default="first")
    pp.set_defaults(func=cmd_play)

    args = p.parse_args()
    if not args.cmd:
        p.print_help(); return
    os.makedirs(args.runs_dir, exist_ok=True)
    args.func(args)

if __name__ == "__main__":
    main()
