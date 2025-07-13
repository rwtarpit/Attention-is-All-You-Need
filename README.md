# Attention-is-All-You-Need
Implementing the ground-breaking paper 'Attention is all you need' from scratch using only Pytorch

This repo contains a from-scratch PyTorch implementation of the legendary paper "Attention Is All You Need" by Vaswani et al. (2017), which introduced the Transformer architecture to the world. Spoiler alert: it changed everything.

🚀 Features
Encoder and Decoder architecture ✅

Multi-head self-attention mechanism ✅

Positional Encoding ✅

Masked Attention for autoregressive decoding ✅

Training loop with teacher forcing ✅


As of now i have implemented the structure of transformer and the training loop.
just to make sure everything is oe the mark, i overfitted the model with 2 samples only. CrossEntropyLoss < 1 gives strong reason to believe everything is going smooth ATM. Inference loop will also be implemented soon.

I will try to train the model on a appropriate dataset once I get good GPU support.
