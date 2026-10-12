"""Train temporal heads on frozen CLIP frame features and test what they actually use.

Questions:
  1. How much does temporal modelling add over averaging frames (and over zero-shot CLIP)?
  2. How many frames are needed?  (k = 1, 2, 4, 8, 16, uniformly sub-sampled)
  3. Does an order-aware head use frame order?  (accuracy when test frames are shuffled or reversed)
All heads are selected on the validation split only and reported on test over 5 seeds.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)


class MeanPool(nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.fc = nn.Linear(d, c)

    def forward(self, x):
        return self.fc(x.mean(1))


class AttnPool(nn.Module):
    """Learned weighting of frames (order-agnostic)."""
    def __init__(self, d, c):
        super().__init__()
        self.score = nn.Linear(d, 1)
        self.fc = nn.Linear(d, c)

    def forward(self, x):
        w = torch.softmax(self.score(x), dim=1)
        return self.fc((w * x).sum(1))


class TemporalConv(nn.Module):
    """1-D convolution over time (order-aware, local)."""
    def __init__(self, d, c, h=256):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(d, h, 3, padding=1), nn.GELU(), nn.Conv1d(h, h, 3, padding=1), nn.GELU())
        self.fc = nn.Linear(h, c)

    def forward(self, x):
        return self.fc(self.conv(x.transpose(1, 2)).mean(-1))


class TemporalTransformer(nn.Module):
    """Two-layer transformer with learned positions (order-aware, global)."""
    def __init__(self, d, c, t_max=16, h=256):
        super().__init__()
        self.inp = nn.Linear(d, h)
        self.pos = nn.Parameter(torch.zeros(1, t_max, h))
        layer = nn.TransformerEncoderLayer(h, 4, 2 * h, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.fc = nn.Linear(h, c)

    def forward(self, x):
        z = self.inp(x) + self.pos[:, : x.shape[1]]
        return self.fc(self.enc(z).mean(1))


HEADS = {"mean_pool": MeanPool, "attn_pool": AttnPool, "temporal_conv": TemporalConv, "temporal_transformer": TemporalTransformer}


def subsample(x, k):
    idx = np.linspace(0, x.shape[1] - 1, k).round().astype(int)
    return x[:, idx]


def train_head(name, xtr, ytr, xva, yva, c, seed, epochs=150):
    torch.manual_seed(seed)
    m = HEADS[name](xtr.shape[-1], c)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    xtr_t, ytr_t = torch.tensor(xtr), torch.tensor(ytr)
    best, best_state = -1, None
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(xtr_t))
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            loss = nn.functional.cross_entropy(m(xtr_t[b]), ytr_t[b], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 4:
            acc = accuracy(m, xva, yva)
            if acc > best:
                best, best_state = acc, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    return m


@torch.no_grad()
def predict(m, x):
    m.eval()
    return m(torch.tensor(x)).softmax(-1).numpy()


def accuracy(m, x, y):
    return float((predict(m, x).argmax(-1) == y).mean())


def ece(probs, y, bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    return float(sum(abs((pred[s] == y[s]).mean() - conf[s].mean()) * s.mean()
                     for lo, hi in zip(edges[:-1], edges[1:]) if (s := (conf > lo) & (conf <= hi)).any()))


def main(npz="clip_ViT-B-32_T16.npz", seeds=(0, 1, 2, 3, 4)):
    d = np.load(ROOT / "data" / npz)
    xtr, ytr, xva, yva, xte, yte = (d[k] for k in ("train_x", "train_y", "val_x", "val_y", "test_x", "test_y"))
    c = len(d["classes"])
    rng = np.random.default_rng(0)
    results = {"zero_shot_clip": {}}
    zs = (xte.mean(1) / np.linalg.norm(xte.mean(1), axis=1, keepdims=True)) @ d["text"].T
    results["zero_shot_clip"] = {"acc": float((zs.argmax(1) == yte).mean()), "macro_f1": float(f1_score(yte, zs.argmax(1), average="macro"))}

    for name in HEADS:
        for k in (1, 2, 4, 8, 16):
            accs, f1s, eces, shuf, rev = [], [], [], [], []
            for s in seeds:
                m = train_head(name, subsample(xtr, k), ytr, subsample(xva, k), yva, c, s)
                xt = subsample(xte, k)
                p = predict(m, xt)
                accs.append(float((p.argmax(1) == yte).mean()))
                f1s.append(float(f1_score(yte, p.argmax(1), average="macro")))
                eces.append(ece(p, yte))
                if k == 16:
                    order = np.stack([rng.permutation(k) for _ in range(len(xt))])
                    shuf.append(accuracy(m, np.take_along_axis(xt, order[..., None], 1), yte))
                    rev.append(accuracy(m, xt[:, ::-1].copy(), yte))
            r = {"acc_mean": np.mean(accs), "acc_std": np.std(accs), "macro_f1": np.mean(f1s), "ece": np.mean(eces)}
            if k == 16:
                r |= {"acc_shuffled_frames": np.mean(shuf), "acc_reversed_frames": np.mean(rev)}
            results.setdefault(name, {})[f"k={k}"] = {a: round(float(b), 4) for a, b in r.items()}
            print(name, k, results[name][f"k={k}"], flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "temporal_heads.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
