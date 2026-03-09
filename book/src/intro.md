# MiniQuant: Build a Quantization Library from Scratch in Rust

This tutorial walks you through implementing a complete INT8 quantization library using pure Rust and ndarray. You'll build everything from scratch: quantization primitives, calibration methods, quantized tensors, and optimized operations.

## What is Quantization?

Quantization reduces model size and speeds up inference by representing weights and activations in lower precision (e.g., INT8 instead of FP32). This is essential for deploying large language models efficiently.

## Learning by Building

This tutorial follows a sequential step-by-step format with:
- **Clear context**: What you're building and why it matters
- **Guided implementation**: Code structure with specific tasks to complete  
- **Immediate validation**: Tests that verify correctness before moving forward
- **Conceptual grounding**: Explanations that connect code to architecture

## What You'll Build

| Step | Component | Key Concept |
|------|-----------|-------------|
| 1-3 | Core Primitives | FP32 → INT8 conversion |
| 4-6 | Calibration | Range detection methods |
| 7 | QuantizedArray | Storage + metadata |
| 8-9 | Matrix Multiplication | INT8 GEMM |
| 10-11 | Pipeline | Full model conversion |

## Prerequisites

- Rust (1.56+)
- Basic knowledge of linear algebra
- Familiarity with Rust's ownership system

## Setup

```bash
# Clone or download this project
cd miniquant

# Run tests to verify everything works
cargo test

# Build the book
cargo install mdbook
mdbook build book

# Serve locally
mdbook serve book
```

## Next Step

Start with [Step 1: Basic Quantization](./step_01.md)
