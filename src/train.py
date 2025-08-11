import os
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from .net import PolicyValueNet
from .mcts import MCTS
from .replay import ReplayBuffer
from .selfplay import play_episode

def save_ckpt(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_ckpt(model, path, map_location="cpu"):
    sd = torch.load(path, map_location=map_location)
    model.load_state_dict(sd)

def train_loop(runs_dir="runs/c4", epochs=10, selfplay_games=500,
               sims=96, device=None, seed=0,
               lr=2e-3, weight_decay=1e-4, value_loss_scale=0.5,
               batch_size=512, replay_capacity=200_000, blocks=6, ch=64):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyValueNet(ch=ch, blocks=blocks).to(device)
    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    buf = ReplayBuffer(capacity=replay_capacity, seed=seed)
    mcts = MCTS(model, device=device)

    # Warmup self-play to seed buffer minimally
    print("Generating initial self-play...")
    with torch.no_grad():
        for _ in tqdm(range(max(32, batch_size // 2))):
            for planes, pi, z in play_episode(mcts, sims_train=sims):
                buf.add(planes, pi, z)

    best_path = os.path.join(runs_dir, "best.pt")
    save_ckpt(model, best_path)

    for epoch in range(1, epochs + 1):
        # Self-play
        print(f"[Epoch {epoch}] Self-play: {selfplay_games} games")
        st = time.time()
        with torch.no_grad():
            for _ in tqdm(range(selfplay_games)):
                for planes, pi, z in play_episode(mcts, sims_train=sims):
                    buf.add(planes, pi, z)
        sp_time = time.time() - st

        # Train
        print(f"[Epoch {epoch}] Train on {len(buf)} samples")
        model.train()
        total_loss = 0.0
        steps = max(1000, len(buf) // batch_size)  # a few passes
        for _ in tqdm(range(steps)):
            planes, pi_t, z_t = buf.sample(batch_size)
            planes = planes.to(device)
            pi_t = pi_t.to(device)
            z_t = z_t.to(device)
            optim.zero_grad()
            pi_logits, v = model(planes)
            loss_pi = F.kl_div(
                input=(pi_logits.log_softmax(dim=-1)),
                target=pi_t,
                reduction="batchmean"
            )
            loss_v = F.mse_loss(v, z_t)
            loss = loss_pi + value_loss_scale * loss_v
            loss.backward()
            optim.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / steps
        print(f"[Epoch {epoch}] done. self-play {sp_time:.1f}s, avg loss {avg_loss:.4f}")

        # Save as best for now (arena gating optional)
        save_ckpt(model, best_path)

    return best_path
