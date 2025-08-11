import numpy as np
import torch

class ReplayBuffer:
    """Simple FIFO replay for (planes, pi, z)."""
    def __init__(self, capacity=200_000, seed=0):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.data_planes = []
        self.data_pi = []
        self.data_z = []

    def add(self, planes, pi, z):
        self.data_planes.append(planes.astype(np.float32))
        self.data_pi.append(pi.astype(np.float32))
        self.data_z.append(np.float32(z))
        if len(self.data_planes) > self.capacity:
            # pop head
            self.data_planes.pop(0)
            self.data_pi.pop(0)
            self.data_z.pop(0)

    def __len__(self): return len(self.data_planes)

    def sample(self, batch_size=512):
        assert len(self) >= batch_size, "Not enough samples"
        idx = self.rng.choice(len(self), size=batch_size, replace=False)
        planes = torch.tensor(np.stack([self.data_planes[i] for i in idx]))
        pi = torch.tensor(np.stack([self.data_pi[i] for i in idx]))
        z = torch.tensor(np.stack([self.data_z[i] for i in idx]))
        return planes, pi, z
