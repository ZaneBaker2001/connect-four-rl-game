import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(x + h)

class PolicyValueNet(nn.Module):
    def __init__(self, ch=64, blocks=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
        )
        self.res = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])

        self.pi_head = nn.Sequential(
            nn.Conv2d(ch, 2, 1), nn.BatchNorm2d(2), nn.ReLU(),
            nn.Flatten(), nn.Linear(2 * 6 * 7, 7)
        )
        self.v_head = nn.Sequential(
            nn.Conv2d(ch, 1, 1), nn.BatchNorm2d(1), nn.ReLU(),
            nn.Flatten(), nn.Linear(6 * 7, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh()
        )

    def forward(self, planes):  # [B,2,6,7]
        x = self.stem(planes)
        x = self.res(x)
        pi = self.pi_head(x)
        v = self.v_head(x).squeeze(-1)
        return pi, v
