# Contributing to TurboQuant 🚀

Thank you for your interest in contributing to TurboQuant! We welcome contributions from the community to make KV cache compression faster and more accessible.

## 🛠️ Development Setup

TurboQuant is a hybrid Rust/Python project. You will need:
- **Rust** (2021 edition)
- **Python** 3.9+
- **uv** (recommended for dependency management)
- **maturin** (for building the Rust-Python bridge)

### 1. Clone and Setup
```bash
git clone https://github.com/swaroski/turboquant.git
cd turboquant
uv venv
source .venv/bin/activate
```

### 2. Install in Editable Mode
This compiles the Rust core and installs the Python package.
```bash
uv pip install -e ".[dev,torch]"
```

## 🧪 Testing

We maintain high standards for both performance and correctness.

### Python Tests
```bash
pytest tests/python
```

### Rust Tests
```bash
cargo test
```

## 🌿 Contribution Workflow

1.  **Fork** the repository.
2.  Create a **Feature Branch** (`git checkout -b feature/amazing-optimization`).
3.  Implement your changes and **add tests**.
4.  Run `ruff check .` to ensure Python style compliance.
5.  **Commit** your changes (`git commit -m 'feat: add SIMD kernels for AVX2'`).
6.  **Push** to the branch (`git push origin feature/amazing-optimization`).
7.  Open a **Pull Request**.

## ⚖️ License
By contributing to TurboQuant, you agree that your contributions will be licensed under its Apache-2.0 License.
