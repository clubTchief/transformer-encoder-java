# Modernized Transformer Encoder Stack in Java (DJL)

A transparent, zero-magic Java implementation of a 6-Layer Stacked Transformer Encoder Pipeline using Deep Java Library (DJL) and PyTorch native execution. 

This repository includes both classic baseline implementations (Vaswani et al., 2017) and modernized production-aligned architectures (RoPE, SwiGLU, Pre-LN) used in frontier LLMs like LLaMA and Mistral.

## Architectural Comparison

| Architectural Feature | Classic Transformer (Vaswani et al. 2017) | Modernized Stack (V8 Implementation) |
| :--- | :--- | :--- |
| **Norm Positioning** | Post-LN: `LayerNorm(X + SubLayer(X))` | Pre-LN: `X + SubLayer(LayerNorm(X))` |
| **Positional Scheme** | Absolute Additive Sinusoidal ($E + P$) | Rotary Position Embeddings (RoPE) |
| **FFN Non-Linearity** | 2-Layer ReLU: $\max(0, X W_1 + b_1) W_2$ | Gated SwiGLU: $(\text{Swish}(X W_g) \odot X W_1) W_2$ |

## Executables Included

1. `TransformerEncoderBlock.java` — Single-block baseline matching Vaswani et al. (2017).
2. `HierarchicalStackedEncoderPipeline.java` — 6-Layer deep stacked baseline with polysemy tracking.
3. `ModernizedHierarchicalEncoderPipeline.java` — Full modern stack (Pre-LN + RoPE + SwiGLU).

## How to Run

Ensure Java 17+ and Maven are installed:

```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/transformer-encoder-java.git
cd transformer-encoder-java

# Build the project
mvn clean compile

# Run the Modernized 6-Layer Pipeline
mvn exec:java -Dexec.mainClass="com.example.math.ModernizedHierarchicalEncoderPipeline"
```

## Documentation & Reference Manuals
Full step-by-step mathematical guides and execution traces are documented in the project Google Docs technical reference manual (V8).