# TurboQuant

High-performance KV cache compression for LLM inference, powered by a Rust core with a clean Python API.

## 🚀 Overview

TurboQuant provides production-grade tools to compress Key-Value (KV) caches in Large Language Models. By employing Fast Walsh-Hadamard Transforms (FWHT) and bit-packed quantization, it allows you to significantly reduce memory usage during autoregressive inference without sacrificing substantial accuracy.

### Key Features
- **4-bit Quantization**: Reduce KV cache memory footprint by ~4x.
- **Outlier Resilience**: FWHT pre-processing balances variance across dimensions.
- **Rust Performance**: Core logic implemented in Rust for maximum efficiency and memory safety.
- **Seamless Integrations**: First-class support for PyTorch and NumPy.
- **Zero-Copy**: Efficient data exchange between Python and Rust.

## 📦 Installation

TurboQuant uses `maturin` for building. It is recommended to use `uv` for management.

```bash
# Clone the repository
git clone https://github.com/your-repo/turboquant
cd turboquant

# Install in editable mode with torch support
uv pip install -e ".[torch]"
```

## 🛠️ Quick Start

### PyTorch API (Recommended)

```python
import torch
import turboquant as tq
from turboquant.torch import TorchKVCache

# Configure the codec
config = tq.KVCodecConfig(
    head_dim=128, 
    key_bits=4, 
    value_bits=4
)

# Initialize the cache
cache = TorchKVCache(config, batch_size=1, num_heads=32)

# Append new KV pairs [B, H, S, D]
keys = torch.randn(1, 32, 1, 128)
values = torch.randn(1, 32, 1, 128)
cache.append(keys, values)

# Compute approximate attention scores [B, H, 1, S]
query = torch.randn(1, 32, 1, 128)
scores = cache.attention_scores(query)

print(f"Cache Tokens: {cache.cache.num_tokens}")
print(f"Scores Shape: {scores.shape}")
```

### Pure Python/NumPy API

```python
import turboquant as tq
import numpy as np

config = tq.KVCodecConfig(head_dim=128, key_bits=4)
codec = tq.KVCodec(config)
cache = codec.create_cache(batch_size=1, num_heads=32)

keys = np.random.randn(1, 32, 8, 128).astype(np.float32)
values = np.random.randn(1, 32, 8, 128).astype(np.float32)

cache.append(keys, values)
```

## 🏗️ Architecture

- **Transform Layer**: Applies FWHT to rotation-invariant data to mitigate the effect of "outlier" features.
- **Quantization Layer**: Symmetric uniform quantization with per-head/per-token scales.
- **Packing Layer**: Bit-level manipulation in Rust for compact storage.
- **Attention Layer**: Approximate inner-product computation directly on compressed representations.

## 🧪 Development

### Running Tests
```bash
# Run Python tests
pytest tests/python

# Run Rust tests
cargo test
```

### Building Wheels
```bash
maturin build --release
```

## ⚖️ License
Apache-2.0
