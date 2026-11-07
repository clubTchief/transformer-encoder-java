package com.example.math;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.List;

public class DecoderBlock {

    private final LayerNorm norm1;
    private final LayerNorm norm2;

    private final NDArray W_Q, W_K, W_V, W_O;
    private final NDArray W_gate, b_gate, W_1, b_1, W_2, b_2;

    private final int dModel;
    private final int numHeads;
    private final int dK;

    public DecoderBlock(NDManager manager, int dModel, int numHeads, int dFF, float eps) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dK = dModel / numHeads;

        // 1. LayerNorms
        this.norm1 = new LayerNorm(manager, dModel, eps);
        this.norm2 = new LayerNorm(manager, dModel, eps);

        // 2. Multi-Head Attention Weights
        float stddevK = (float) Math.sqrt(2.0 / (dModel + dK));
        float stddevO = (float) Math.sqrt(2.0 / (dModel + dModel));

        this.W_Q = createWeight(manager, new Shape(dModel, dModel), stddevK);
        this.W_K = createWeight(manager, new Shape(dModel, dModel), stddevK);
        this.W_V = createWeight(manager, new Shape(dModel, dModel), stddevK);
        this.W_O = createWeight(manager, new Shape(dModel, dModel), stddevO);

        // 3. SwiGLU FFN Weights & Biases
        float stddev1 = (float) Math.sqrt(2.0 / (dModel + dFF));
        float stddev2 = (float) Math.sqrt(2.0 / (dFF + dModel));

        this.W_gate = createWeight(manager, new Shape(dModel, dFF), stddev1);
        this.b_gate = createBias(manager, new Shape(dFF));
        this.W_1 = createWeight(manager, new Shape(dModel, dFF), stddev1);
        this.b_1 = createBias(manager, new Shape(dFF));
        this.W_2 = createWeight(manager, new Shape(dFF, dModel), stddev2);
        this.b_2 = createBias(manager, new Shape(dModel));
    }

    /**
     * Holds the block's output tensor alongside its updated K and V cache states.
     */
    public static class DecoderBlockResult {
        public final NDArray output;
        public final NDArray kCache;
        public final NDArray vCache;

        public DecoderBlockResult(NDArray output, NDArray kCache, NDArray vCache) {
            this.output = output;
            this.kCache = kCache;
            this.vCache = vCache;
        }
    }

    /**
     * Standard Non-Cached Forward Pass (used during full sequence training).
     */
    public NDArray forward(NDArray X, NDArray causalMask, NDArray cos, NDArray sin) {
        return forwardWithCache(X, causalMask, cos, sin, null, null).output;
    }

    /**
     * Forward pass supporting Layer-level KV-Caching for O(1) step inference.
     */
    public DecoderBlockResult forwardWithCache(NDArray X, NDArray causalMask, NDArray cos, NDArray sin, NDArray kCache, NDArray vCache) {
        long seqLen = X.getShape().get(0);

        // --- SubLayer 1: Pre-LN Causal Multi-Head Attention ---
        NDArray normX1 = norm1.forward(X);

        NDArray Q = normX1.matMul(W_Q);
        NDArray K = normX1.matMul(W_K);
        NDArray V = normX1.matMul(W_V);

        NDArray Q_split = Q.reshape(new Shape(seqLen, numHeads, dK)).transpose(1, 0, 2);
        NDArray K_split = K.reshape(new Shape(seqLen, numHeads, dK)).transpose(1, 0, 2);
        NDArray V_split = V.reshape(new Shape(seqLen, numHeads, dK)).transpose(1, 0, 2);

        // Apply pre-computed RoPE cos/sin matrices to new token projections
        NDArray Q_rope = ModernizedHierarchicalEncoderPipeline.applyRoPE(Q_split, cos, sin);
        NDArray K_rope = ModernizedHierarchicalEncoderPipeline.applyRoPE(K_split, cos, sin);

        // Concatenate new Keys and Values with incoming KV-Cache along sequence dimension (dim=1)
        if (kCache != null && vCache != null) {
            K_rope = kCache.concat(K_rope, 1);
            V_split = vCache.concat(V_split, 1);
        }

        NDArray newKCache = K_rope;
        NDArray newVCache = V_split;

        long keySeqLen = K_rope.getShape().get(1);

        // Scaled Dot-Product Attention
        NDArray rawScores = Q_rope.matMul(K_rope.transpose(0, 2, 1)).div(Math.sqrt(dK));

        if (causalMask != null) {
            // Align mask dimensions if shapes match slice requirements
            if (causalMask.getShape().get(0) == seqLen && causalMask.getShape().get(1) == keySeqLen) {
                rawScores = rawScores.add(causalMask);
            }
        }

        NDArray attentionWeights = rawScores.softmax(-1);
        NDArray headOutputs = attentionWeights.matMul(V_split);
        NDArray concatOutputs = headOutputs.transpose(1, 0, 2).reshape(new Shape(seqLen, dModel));
        NDArray mhaOutput = concatOutputs.matMul(W_O);

        NDArray subLayer1Out = X.add(mhaOutput);

        // --- SubLayer 2: Pre-LN SwiGLU Feed-Forward Network ---
        NDArray normX2 = norm2.forward(subLayer1Out);

        NDArray gateLinear = normX2.matMul(W_gate).add(b_gate);
        NDArray gateSwish = ModernizedHierarchicalEncoderPipeline.applySwish(gateLinear);
        NDArray featureLinear = normX2.matMul(W_1).add(b_1);
        NDArray ffnHidden = gateSwish.mul(featureLinear);
        NDArray ffnOutput = ffnHidden.matMul(W_2).add(b_2);

        NDArray finalOutput = subLayer1Out.add(ffnOutput);

        return new DecoderBlockResult(finalOutput, newKCache, newVCache);
    }

    /**
     * Returns all registered learnable parameters in this block for optimization.
     */
    public List<NDArray> params() {
        List<NDArray> parameters = new ArrayList<>();
        parameters.add(norm1.getGamma());
        parameters.add(norm1.getBeta());
        parameters.add(norm2.getGamma());
        parameters.add(norm2.getBeta());
        parameters.addAll(List.of(W_Q, W_K, W_V, W_O, W_gate, b_gate, W_1, b_1, W_2, b_2));
        return parameters;
    }

    private NDArray createWeight(NDManager manager, Shape shape, float stddev) {
        NDArray weight = manager.randomNormal(0.0f, stddev, shape, DataType.FLOAT32);
        weight.setRequiresGradient(true);
        return weight;
    }

    private NDArray createBias(NDManager manager, Shape shape) {
        NDArray bias = manager.zeros(shape, DataType.FLOAT32);
        bias.setRequiresGradient(true);
        return bias;
    }
}