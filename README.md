# 🐈 Continuous Attention Transformer (CAT)

This repository contains a reference implementation of the **Continuous Attention Transformer (CAT)**.

CAT replaces traditional transformer layers with **continuous-attention neural ODE blocks**, enabling smooth information flow over arbitrary sequence lengths with minimal parameter overhead. Attention kernels are parameterized using a **low-rank spectral basis** (Fourier or Chebyshev) and regularized using a **Sobolev penalty** to encourage smoothness.

---

## 🔬 Core Features

- Continuous-depth attention via Neural ODEs.
- Learned row-stochastic attention kernels over continuous time.
- Basis options: `fourier` or `cheb` (Chebyshev polynomials).
- Works on CPU, CUDA, and Apple Silicon (MPS).
- Lightweight Tiny-Shakespeare benchmark (~50k characters).
- Optional baseline Transformer for comparison.

---

## 📦 Installation

```bash
pip install torch numpy torchdiffeq
```

Optional (if not already available):
```bash
pip install --upgrade pip setuptools
```

---

## 🚀 Quick Start

Run the CAT model with Chebyshev basis on MPS:

```bash
python CAT.py \
    --device mps \
    --basis cheb \
    --blocks 3 \
    --train_steps 2000 \
    --d_model 256 \
    --n_heads 8 \
    --n_freq 48 \
    --sob_lambda 5e-7
```

---

## 📊 Outputs

During training, the script prints:
- Loss comparison between CAT and baseline.
- Final validation metrics on Tiny-Shakespeare:
  - Cross-Entropy (CE)
  - Perplexity (PPL)
  - Evaluation at multiple context lengths (128–5000 tokens)

---

## 🧠 Architecture Highlights

- CAT stacks ODE blocks, each integrating a continuous-attention layer.
- Each kernel is a spectral expansion in a compact 2D basis.
- Sobolev penalty discourages over-oscillatory kernels.
- Training includes annealed regularization and fast Legendre quadrature.

---

## 📄 Notes

- This code is meant for demonstration and reproducibility of early results.
- The idea of multiple ODE blocks remains relevant: even if a single functional layer can model the full sequence, multiple blocks help guide complex feature flow.
- Context length is effectively unbounded — limited only by hardware memory.

---

## 🔗 License

MIT License.  
(c) 2025 Umberto Puddu
