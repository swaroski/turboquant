# TurboQuant 🚀

[![Rust](https://img.shields.io/badge/core-rust-orange?logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/api-python-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

**TurboQuant** is a high-performance KV cache compression library for LLM inference. It combines a SIMD-optimized Rust backend with Fast Walsh-Hadamard Transforms (FWHT) to enable aggressive 4-bit quantization with minimal accuracy loss.

---

## 💡 Why TurboQuant?

As sequence lengths grow, the KV cache becomes the primary memory bottleneck in LLM serving. TurboQuant solves this by:

1.  **Outlier Mitigation**: Applies FWHT to "rotate" the feature space, spreading high-magnitude outliers across dimensions before quantization.
2.  **4-bit Quantization**: Reduces memory footprint by **4x** compared to FP16, allowing for larger batches and longer contexts.
3.  **Rust Core**: Handles bit-packing and decompression at the memory limit, bypassing Python's performance overhead.

### Memory Savings
| Component | Original (FP16) | TurboQuant (4-bit) | Savings |
| :--- | :--- | :--- | :--- |
| **Key Cache** | 2 bytes / elem | 0.5 bytes / elem | **75%** |
| **Value Cache** | 2 bytes / elem | 0.5 bytes / elem | **75%** |

---

## 📦 Installation

TurboQuant is built with `maturin` and managed with `uv`.

```bash
# Install with PyTorch support
uv pip install -e ".[torch]"
```

---

## 🛠️ Usage

### PyTorch Integration

The `TorchKVCache` wrapper provides a seamless interface for existing transformer models.

```python
import torch
import turboquant as tq
from turboquant.torch import TorchKVCache

# 1. Configure the Codec
config = tq.KVCodecConfig(head_dim=128, key_bits=4, value_bits=4)

# 2. Initialize the Cache [Batch, Heads]
cache = TorchKVCache(config, batch_size=1, num_heads=32)

# 3. Append Tokens [B, H, Seq, Dim]
k, v = torch.randn(1, 32, 1, 128), torch.randn(1, 32, 1, 128)
cache.append(k, v)

# 4. Compute Approximate Scores
q = torch.randn(1, 32, 1, 128)
scores = cache.attention_scores(q) # Returns approximate scores [1, 32, 1, 1]
```

---

## 🏗️ Architecture

- **`src/core/rotation.rs`**: SIMD-accelerated Fast Walsh-Hadamard Transform.
- **`src/core/quantizer.rs`**: Per-token symmetric uniform 4-bit quantization.
- **`src/core/attention.rs`**: Optimized inner-product kernels for compressed representations.
- **`turboquant/torch/`**: PyTorch ops and tensor handling.

---

## ⚖️ License
Licensed under Apache-2.0.
