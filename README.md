# MiniQuant

Build a quantization library from scratch in Rust.

This tutorial walks you through implementing a complete INT8 quantization library using pure Rust and ndarray. Learn quantization by building it yourself.

## Quick Start

```bash
# Run all tests
cargo test

# Run specific step tests
cargo test core::quantize

# Build the book
cargo install mdbook
mdbook build book

# Serve locally
mdbook serve book
```

## Tutorial Steps

| Step | Topic |
|------|-------|
| 1 | Basic Quantization |
| 2 | Scale Calculation |
| 3 | Symmetric vs Asymmetric |
| 4 | Min-Max Calibration |
| 5 | Percentile Calibration |
| 6 | Dynamic Quantization |
| 7 | QuantizedArray |
| 8 | Quantized MatMul |
| 9 | Operation Fusion |
| 10 | Observer Pattern |
| 11 | Model Converter |

## Project Structure

```
miniquant/
├── src/
│   ├── core/           # Steps 1-3
│   ├── calibration/    # Steps 4-6
│   ├── tensor/        # Step 7
│   ├── ops/           # Steps 8-9
│   └── pipeline/      # Steps 10-11
├── book/              # mdBook tutorial
└── Cargo.toml
```

## License

MIT
