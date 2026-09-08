# ML From Scratch

> **Session type:** build sessions, Python. Start with: *"Read the README, let's start phase 1."*
> Above-the-line rule applies to the math and the architecture. Below the line: library syntax,
> CUDA/driver setup, plotting.

## What it is
Build and train neural networks **from first principles**, ending with a small language model
trained overnight on this laptop. Not calling an API — building the thing.

## Hardware (verified 2026-08-30)
- **GPU:** NVIDIA RTX 4070 Laptop, **8 GB VRAM**, driver 616.56 — CUDA-capable
- **CPU:** i7-13700HX (16 cores / 24 threads), **32 GB RAM**
- **Python:** 3.12.10

**What that hardware can actually do:**
- ✅ MNIST / CIFAR CNNs — minutes
- ✅ Character-level transformer, ~10–50M params — **trains overnight, comfortably**
- ✅ LoRA fine-tune of a 1–3B model (quantized) — hours
- ❌ Training a 7B+ model from scratch — not on 8 GB, don't try

## Why this project
- **Differentiator.** Most students can call an LLM API. Very few can explain backprop because
  they *implemented* it. That gap shows instantly in interviews.
- **It's the honest version of "I know AI."** Ties to the stated interest in AI without being
  another agent-orchestration project.
- **Portfolio value:** a from-scratch neural net + a trained-it-myself language model is a
  genuinely strong pair for a junior CE resume.

## Roadmap

**Phase 1 — neural net in raw NumPy (no PyTorch).** MNIST digit classifier. Implement forward pass,
loss, **backprop by hand**, and gradient descent. Runs on CPU in minutes.
*This phase is the whole point — do not skip to PyTorch.* If you can derive and code backprop,
you understand ML. If you can't, you're an API caller.

**Phase 2 — same thing in PyTorch.** Rebuild phase 1 using a real framework. The contrast is the
lesson: see exactly what `loss.backward()` was doing for you.

**Phase 3 — CNN on CIFAR-10.** Convolutions, pooling, data augmentation. First real GPU training.

**Phase 4 — character-level transformer, trained overnight.** Implement attention from scratch,
train on a text corpus you pick (something with personality — not generic scraped web text).
This is the headline portfolio piece. Log the training curve; the artifact is the *process*,
not just the weights.

**Phase 5 — write it up.** A README with training curves, sample outputs, what failed, what you'd
change. The write-up is a large share of the portfolio value.

## Phase 1 first slice
1. Load MNIST (`torchvision` or raw idx files — your call).
2. One layer, forward pass only: `output = input @ weights + bias`. Verify the shapes.
3. Then loss. Then the backward pass, one layer at a time.

Design questions that are yours:
- What are the matrix shapes at each step, and why? *(Shape errors will be 90% of your bugs —
  learn to reason about them, not guess.)*
- Why does the loss need to be differentiable?
- What actually is a gradient, in one plain sentence?

## Setup notes (below the line, just do these)
- Use a virtual environment: `python -m venv .venv`
- Install the **CUDA** build of PyTorch, not the CPU one — check `torch.cuda.is_available()`
  returns `True` before phase 3, or you'll silently train on CPU and wonder why it takes 40 hours.
- Phases 1–2 need only `numpy` and `matplotlib`.
