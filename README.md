# Transformer Block from Scratch (Pure NumPy)

> **⚠️ Work in Progress:** This repository is actively being built. The current files represent partial milestones towards a complete Transformer Block.

A pure NumPy implementation of multi-head attention with complete manual backpropagation, including the derivation of the softmax Jacobian and gradient flow through causal masks.

## Overview
Modern deep learning frameworks like PyTorch and TensorFlow abstract away the complex calculus and gradient flow of neural networks. While `import torch.nn as nn` is standard for production, relying exclusively on auto-grad engines can leave engineers with a shallow understanding of the underlying architecture.

The goal of this "toy model" is to prove fundamental comprehension of the linear algebra and multivariable calculus that powers modern Large Language Models (LLMs). This repository documents the evolution of a Neural Network from a single-head attention mechanism into a fully functional GPT Transformer block, written entirely from scratch without any ML libraries.

## Phase 1: The Single-Head Foundation
The initial implementation (`01_single_head.py`) focused on proving the core calculus of Self-Attention:
1. **The Softmax Jacobian:** Deriving the complex Jacobian gradient of the Softmax function into a fully vectorized NumPy operation: `dZ_i = S_i (E_i - sum_j E_j S_j)`.
2. **Matrix Transposition for Gradient Flow:** Explicitly demonstrating why transposed matrices are required during the backward pass (e.g., `dW_q = sentence_embedding.T @ dQ`) to map blame back to original input features.
3. **The Total Derivative Rule:** Tracing upstream gradients from Queries, Keys, and Values back into a single unified gradient for the `sentence_embedding`.

## Phase 2: Scaling to a GPT Architecture
The secondary implementation (`02_multi_heads.py`) upgrades the foundation into a true Generative Transformer Block by introducing parallel processing and time-awareness:
1. **Multi-Head Attention:** Splitting the embedding dimension into parallel "brains" to learn specialized syntactic and semantic relationships (e.g., Head 1 focusing on Position, Head 2 focusing on Nouns). The gradients are manually sliced and recombined along the feature axis (`axis=-1`) during backpropagation.
2. **Learned Positional Encodings:** Mimicking OpenAI's GPT design philosophy by relying on pure gradient descent to learn absolute position (`E[token] + P`), rather than hardcoding static Sine/Cosine formulas.
3. **Causal Masking (The Blindfold):** Generating an upper-triangular matrix (`np.triu`) of Negative Infinity (`-1e9`) to prevent the AI from "cheating" by looking into the future. This mathematically forces the Softmax function (`e^-infinity`) to deterministically crush future probabilities to `0.000`.

## Architecture Summary
The final Neural Network consists of:
* Trainable Vocabulary & Positional Embedding Matrices (`E`, `P`)
* A Causal Mask (`-1e9` upper triangle)
* A Multi-Head Self-Attention Layer (`W_q`, `W_k`, `W_v`)
* A Feed-Forward Hidden Layer (`W1`, `b1`)
* A Softmax Output Layer (`W2`, `b2`) trained via Categorical Cross-Entropy.

## Usage
The architecture is split into heavily commented, pedagogical steps to maximize readability. 
   
To run the Phase 1 training loop:
```bash
python core/01_single_head.py
```

To run the Phase 2 training loop (Full Transformer):
```bash
python core/02_multi_heads.py
```
