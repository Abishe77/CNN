# CNN from Scratch — Fashion MNIST

A Convolutional Neural Network built **entirely from scratch using NumPy** — no PyTorch, no TensorFlow, no deep learning libraries. Every operation, forward pass, and backpropagation step is implemented manually, including the convolution kernel gradient update.

Trained and tested on a 400-image subset of Fashion-MNIST, achieving **64% test accuracy with a single learnable filter.**

---

## Architecture

```
Input (28×28) → Conv (3×3, 1 filter) → ReLU → MaxPool (2×2) → Flatten → FC(196→64) → ReLU → FC(64→10) → Softmax
```

| Stage | Details |
|---|---|
| Input | 28×28 grayscale image |
| Convolution | 1 filter of size 3×3, padding=1, stride=1 |
| Activation | ReLU |
| Pooling | Max Pooling, 2×2, stride=2 → output 14×14 |
| Flatten | 14×14 = 196-dimensional vector |
| Hidden Layer | 196 → 64 neurons, ReLU |
| Output Layer | 64 → 10 neurons, Softmax |
| Loss | Categorical Cross-Entropy |

---

## What's Implemented from Scratch

- **Convolution** — sliding kernel over padded image, element-wise dot product accumulation
- **ReLU** — forward and backward (gradient masking via `fmatrix > 0`)
- **Max Pooling** — forward and backward (gradient routed only to max position via mask)
- **Softmax** — numerically stable via max-shift trick
- **Cross-Entropy Loss** — with epsilon stabilization to avoid `log(0)`
- **Backpropagation** — full gradient flow through:
  - Dense output layer
  - Dense hidden layer
  - Un-pooling (max-mask gradient routing)
  - ReLU gate
  - Convolution filter (`d_filter` accumulated over batch)
- **Gradient Descent** — vanilla SGD updating filter + all weights and biases

---

## Dataset

**Fashion-MNIST** — 10 clothing categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

Only 400 images are used (subset for speed):
- 200 training, 200 testing
- Images are shuffled before splitting to avoid class ordering bias
- Normalized to [0, 1] by dividing pixel values by 255

---

## Results

| Metric | Value |
|---|---|
| Training subset | 200 images |
| Test subset | 200 images |
| Filters used | 1 |
| Test Accuracy | **64%** |
| Train Accuracy | **73.5%** |
| Epochs | 50 |
| Learning Rate | 0.25 |

> 64% accuracy with a single filter and 200 training samples is a meaningful result — the network is genuinely learning spatial features, not guessing (random baseline = 10%).

---

## How to Run

**Requirements**
```
numpy
matplotlib
scikit-learn
```

Install:
```bash
pip install numpy matplotlib scikit-learn
```

Run:
```bash
python cnn_from_scratch.py
```

Training prints loss and accuracy every epoch. After 50 epochs, test accuracy is printed.

---

## Limitations & What's Next

- Only **1 filter** — adding multiple filters per layer would capture richer features
- **Small subset** (200 train) — scaling to full 60,000 images would significantly boost accuracy
- **No momentum or adaptive learning rate** — vanilla SGD only; Adam would converge faster
- **Single conv layer** — stacking conv layers would allow hierarchical feature learning
- Slow due to **pure Python loops** in convolution and pooling — can be vectorized with `np.lib.stride_tricks`

---


