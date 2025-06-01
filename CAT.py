#!/usr/bin/env python
# CAT.py — Continuous Attention Transformer 🐈
#
# • Continuous‐attention ODE blocks (1 … K stacked)
# • Choice of spectral basis →  `--basis {fourier,cheb}`
# • Tiny-Shakespeare demo prints CE / PPL up to 5 000 tokens
# • Works on CPU, CUDA or Apple-Silicon (MPS)
#
# Example:
#   python CAT.py \
#       --device mps \
#       --basis fourier \
#       --blocks 3 \
#       --train_steps 2000 \
#       --d_model 256 \
#       --n_heads 8 \
#       --n_freq 48 \
#       --sob_lambda 5e-7

from __future__ import annotations
import math, os, urllib.request, argparse, warnings, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

try:
    from torchdiffeq import odeint_adjoint as odeint  # GPU‐friendly
except ImportError:
    from torchdiffeq import odeint  # CPU fallback

# ─── Spectral bases on [0,1] ─────────────────────────────────────────────

def fourier_basis(n: int, t: torch.Tensor) -> torch.Tensor:
    k = torch.arange(n, device=t.device).unsqueeze(1)
    return torch.cos(2 * math.pi * k * t.unsqueeze(0)) / math.sqrt(n)

def cheb_basis(n: int, t: torch.Tensor) -> torch.Tensor:
    x = 2 * t - 1  # map to [–1,1]
    T = torch.ones((n, len(t)), device=t.device)
    if n > 1:
        T[1] = x
    for i in range(2, n):
        T[i] = 2 * x * T[i - 1] - T[i - 2]
    return T / math.sqrt(n)

# ─── 2D Kernel with Pluggable Basis ──────────────────────────────────────

class Kernel2D(nn.Module):
    def __init__(self, n_freq: int, basis: str = "fourier"):
        super().__init__()
        self.n = n_freq
        self.basis = basis  # "fourier" or "cheb"
        self.coeffs = nn.Parameter(0.01 * torch.randn(n_freq, n_freq))
        self.log_scale = nn.Parameter(torch.zeros(1))

    def _basis(self, t: torch.Tensor) -> torch.Tensor:
        return fourier_basis(self.n, t) if (self.basis == "fourier") else cheb_basis(self.n, t)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        Φs = self._basis(s)  # [n, S]
        Φt = self._basis(t)  # [n, T]
        W = Φs.t() @ self.coeffs @ Φt  # [S, T]
        W = F.softplus(self.log_scale) * W
        return torch.softmax(W, dim=-1)  # row‐stochastic

    def sobolev_penalty(self) -> torch.Tensor:
        idx = torch.arange(self.n, device=self.coeffs.device)
        lap4 = (2 * math.pi) ** 4 * (idx[:, None] ** 2 + idx[None] ** 2) ** 2
        return (lap4 * self.coeffs ** 2).sum()

# ─── Continuous‐Attention Layer ─────────────────────────────────────────

class ContinuousAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_freq: int,
        basis: str,
        quad: int = 64,
        drop: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d = d_model // n_heads
        self.kernels = nn.ModuleList([Kernel2D(n_freq, basis) for _ in range(n_heads)])
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(drop)

        nodes, w = np.polynomial.legendre.leggauss(quad)
        self.register_buffer("t_nodes", torch.tensor((nodes + 1) / 2, dtype=torch.float32))
        self.register_buffer("t_w", torch.tensor(w / 2, dtype=torch.float32))

    def sobolev_penalty(self) -> torch.Tensor:
        return sum(k.sobolev_penalty() for k in self.kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, _ = x.shape
        s = torch.linspace(0, 1, T, device=x.device)  # [T]

        # project and reshape: V ∈ [h, B, T, (d_model/h)]
        V = self.v_proj(x).view(B, T, self.h, self.d).permute(2, 0, 1, 3)

        idx_f = self.t_nodes * (T - 1)
        idx = idx_f.long()
        frac = (idx_f - idx).view(1, -1, 1, 1)  # [1, Q, 1, 1]

        V0 = V[:, :, idx, :].permute(0, 2, 1, 3)  # [h, Q, B, d]
        V1 = V[:, :, idx + 1, :].permute(0, 2, 1, 3)
        Vq = (1 - frac) * V0 + frac * V1  # [h, Q, B, d]

        tw = self.t_w  # [Q]
        outs = []
        for h_idx, ker in enumerate(self.kernels):
            W = ker(s, self.t_nodes)  # [T, Q]
            out = torch.einsum("tq, qbd -> tbd", W * tw, Vq[h_idx])  # [T, B, d]
            outs.append(out)

        # concatenate heads and project back [B, T, D]
        out = torch.cat(outs, dim=-1).permute(1, 0, 2)
        return self.dropout(self.out_proj(out))

# ─── Neural‐ODE Block (1 “layer” of CAT) ─────────────────────────────────

class ODEFunc(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_freq: int, basis: str):
        super().__init__()
        self.attn = ContinuousAttention(d_model, n_heads, n_freq, basis)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, τ: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # h: [T, B, D]
        h2 = h.permute(1, 0, 2)  # [B, T, D]
        a = self.attn(h2).permute(1, 0, 2)  # [T, B, D]
        h = h + a
        return self.ff(self.norm(h))

# ─── Depth‐Integrator Helper ────────────────────────────────────────────

def integrate(block: ODEFunc, y0: torch.Tensor, dev: str) -> torch.Tensor:
    # y0: [T, B, D]
    if dev == "mps":
        # 4‐stage RK4 in float32
        dt, h = 0.25, y0
        for _ in range(4):
            k1 = block(None, h)
            k2 = block(None, h + 0.5 * dt * k1)
            k3 = block(None, h + 0.5 * dt * k2)
            k4 = block(None, h + dt * k3)
            h = h + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return h

    τ = torch.tensor([0.0, 1.0], device=y0.device)
    return odeint(block, y0, τ, method="dopri5")[1]  # return h(1)

# ─── CAT Model (K Stacked ODE Blocks) ───────────────────────────────────

class CAT(nn.Module):
    def __init__(
        self,
        vocab: int,
        d_model: int,
        n_heads: int,
        n_freq: int,
        sobλ: float,
        blocks: int,
        basis: str,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList(
            [ODEFunc(d_model, n_heads, n_freq, basis) for _ in range(blocks)]
        )
        self.proj = nn.Linear(d_model, vocab)
        self.sobλ = sobλ
        self.vocab = vocab

    def forward(self, tok: torch.Tensor, *, penalty: bool = False):
        # tok: [B, T]
        h = self.embed(tok).transpose(0, 1)  # [T, B, D]
        for blk in self.blocks:
            h = integrate(blk, h, tok.device.type)
        logits = self.proj(h.transpose(0, 1))  # [B, T, V]

        sob = sum(b.attn.sobolev_penalty() for b in self.blocks)
        return logits, (self.sobλ * sob if penalty else torch.tensor(0.0, device=tok.device))

# ─── Baseline Transformer (1 layer for fair comparison) ────────────────

class Baseline(nn.Module):
    def __init__(self, vocab: int, d_model: int, n_heads: int, layers: int, drop: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout=drop, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, layers)
        self.embed = nn.Embedding(vocab, d_model)
        self.proj = nn.Linear(d_model, vocab)
        self.drop = nn.Dropout(drop)
        self.vocab = vocab

    def forward(self, tok: torch.Tensor, *, penalty: bool = False):
        h = self.enc(self.embed(tok))  # [B, T, D]
        return self.drop(self.proj(h)), torch.tensor(0.0, device=tok.device)

# ─── Data Loader & Evaluator (Tiny–Shakespeare) ─────────────────────────

def load_shakespeare() -> str:
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists("tiny.txt"):
        urllib.request.urlretrieve(url, "tiny.txt")
    return open("tiny.txt", "r", encoding="utf-8").read()

def vocab_encode(text: str) -> tuple[np.ndarray, int]:
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    arr = np.array([stoi[c] for c in text], dtype=np.int64)
    return arr, len(vocab)

@torch.inference_mode()
def evaluate(
    model: nn.Module,
    ids: np.ndarray,
    L: int,
    dev: str,
    max_w: int,
) -> tuple[float, float]:
    total_nll, n_tok = 0.0, 0
    step = L
    n_chunks = (len(ids) - L - 1) // step
    if max_w > 0:
        n_chunks = min(n_chunks, max_w)

    for k in range(n_chunks):
        i = k * step
        x = torch.from_numpy(ids[i : i + L]).unsqueeze(0).to(dev)  # [1, L]
        y = torch.from_numpy(ids[i + 1 : i + L + 1]).unsqueeze(0).to(dev)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab), y.reshape(-1), reduction="sum")
        total_nll += loss.item()
        n_tok += L

    ce = total_nll / n_tok
    return ce, math.exp(ce)

# ─── Main Training + Validation Loop ────────────────────────────────────

def main(cfg):
    # Load data
    text = load_shakespeare()
    ids, V = vocab_encode(text)
    train_ids, val_ids = ids[:50_000], ids[50_000:]

    # Instantiate models
    cat = CAT(V, cfg.d_model, cfg.n_heads, cfg.n_freq, cfg.sob_lambda, cfg.blocks, cfg.basis).to(cfg.device)
    base = Baseline(V, cfg.d_model, cfg.n_heads, cfg.blocks).to(cfg.device)

    optc = torch.optim.AdamW(cat.parameters(), lr=cfg.lr)
    optb = torch.optim.AdamW(base.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(0)

    print(f"Training {cfg.train_steps} mini-batches (crop {cfg.seq}) …")
    for step in range(cfg.train_steps):
        # Random crop of length cfg.seq + 1
        idx = rng.choice(len(train_ids) - cfg.seq - 1, size=cfg.batch)
        batch = np.stack([train_ids[i : i + cfg.seq + 1] for i in idx])
        x = torch.from_numpy(batch[:, :-1]).to(cfg.device)  # [B, seq]
        y = torch.from_numpy(batch[:, 1:]).to(cfg.device)

        # --- CAT step ---
        optc.zero_grad()
        logits_c, sob = cat(x, penalty=True)
        pen_w = cfg.sob_lambda * max(0.2, 1 - step / cfg.train_steps)
        loss_c = loss_fn(logits_c.reshape(-1, V), y.reshape(-1)) + pen_w * sob
        loss_c.backward()
        optc.step()

        # --- Baseline step ---
        optb.zero_grad()
        logits_b, _ = base(x)
        loss_b = loss_fn(logits_b.reshape(-1, V), y.reshape(-1))
        loss_b.backward()
        optb.step()

        if step % 100 == 0:
            print(f"step {step:4d} | CAT {loss_c.item():.3f} | base {loss_b.item():.3f}")

    # Validation summary
    print("\n" + "-" * 62)
    print("Validation summary")
    print("-" * 62)
    print(f"{'len':>5} | CAT CE | PPL | Bas CE | PPL | ΔCE")
    print("-" * 62)
    for L in (128, 512, 1024, 5000):
        ce_c, ppl_c = evaluate(cat, val_ids, L, cfg.device, cfg.eval_windows)
        ce_b, ppl_b = evaluate(base, val_ids, L, cfg.device, cfg.eval_windows)
        print(f"{L:5d} | {ce_c:6.3f} |{ppl_c:5.1f}| {ce_b:6.3f}|{ppl_b:5.1f}| {ce_b - ce_c:+.3f}")
    print("-" * 62)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    # model hyperparameters
    p.add_argument("--blocks", type=int, default=1, help="Number of ODE blocks")
    p.add_argument("--d_model", type=int, default=128, help="Embedding dimension")
    p.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    p.add_argument("--n_freq", type=int, default=32, help="Spectral frequencies per axis")
    p.add_argument(
        "--basis",
        choices=["fourier", "cheb"],
        default="fourier",
        help="Basis for the 2D kernel (fourier or cheb)",
    )
    p.add_argument("--sob_lambda", type=float, default=5e-7, help="Sobolev regulariser weight")
    # optimization
    p.add_argument("--train_steps", type=int, default=800, help="Total training mini‐batches")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--batch", type=int, default=32, help="Batch size")
    p.add_argument("--seq", type=int, default=256, help="Sequence crop length")
    # evaluation
    p.add_argument("--eval_windows", type=int, default=256, help="Max validation windows per length")
    main(p.parse_args())
