package com.example.math;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.List;

public class MultiHeadAttention {

    private final NDArray wQ, wK, wV, wO;
    private final int numHeads;
    private final int dModel;
    private final int headDim;

    public MultiHeadAttention(NDManager manager, int dModel, int numHeads) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.headDim = dModel / numHeads;

        float std = (float) Math.sqrt(2.0 / dModel);
        this.wQ = manager.randomNormal(0.0f, std, new Shape(dModel, dModel), DataType.FLOAT32);
        this.wK = manager.randomNormal(0.0f, std, new Shape(dModel, dModel), DataType.FLOAT32);
        this.wV = manager.randomNormal(0.0f, std, new Shape(dModel, dModel), DataType.FLOAT32);
        this.wO = manager.randomNormal(0.0f, std, new Shape(dModel, dModel), DataType.FLOAT32);

        this.wQ.setRequiresGradient(true);
        this.wK.setRequiresGradient(true);
        this.wV.setRequiresGradient(true);
        this.wO.setRequiresGradient(true);
    }

    public List<NDArray> params() {
        return List.of(wQ, wK, wV, wO);
    }

    /**
     * Container holding updated output tensor and updated KV-Cache states.
     */
    public static class AttentionResult {
        public final NDArray output;
        public final NDArray kCache;
        public final NDArray vCache;

        public AttentionResult(NDArray output, NDArray kCache, NDArray vCache) {
            this.output = output;
            this.kCache = kCache;
            this.vCache = vCache;
        }
    }

    /**
     * Standard Forward Pass (used during batch training).
     */
    public NDArray forward(NDManager manager, NDArray x, NDArray causalMask) {
        return forwardWithCache(manager, x, causalMask, null, null).output;
    }

    /**
     * Forward pass with optional KV-Caching for O(1) step inference.
     */
    public AttentionResult forwardWithCache(NDManager manager, NDArray x, NDArray causalMask, NDArray kCache, NDArray vCache) {
        long seqLen = x.getShape().get(0);

        // Linear projections
        NDArray Q = x.matMul(wQ);
        NDArray K = x.matMul(wK);
        NDArray V = x.matMul(wV);

        // Reshape to Multi-Head: [seqLen, numHeads, headDim] -> transpose to [numHeads, seqLen, headDim]
        Q = Q.reshape(seqLen, numHeads, headDim).transpose(1, 0, 2);
        K = K.reshape(seqLen, numHeads, headDim).transpose(1, 0, 2);
        V = V.reshape(seqLen, numHeads, headDim).transpose(1, 0, 2);

        // Concatenate with past cached Keys and Values if present
        if (kCache != null && vCache != null) {
            K = kCache.concat(K, 1); // Concat along sequence dimension (dim=1)
            V = vCache.concat(V, 1);
        }

        // Updated cache states
        NDArray newKCache = K;
        NDArray newVCache = V;

        long keySeqLen = K.getShape().get(1);

        // Scaled Dot-Product Attention: Q @ K^T / sqrt(headDim)
        NDArray K_T = K.transpose(0, 2, 1);
        NDArray scores = Q.matMul(K_T).div((float) Math.sqrt(headDim));

        // Apply Causal Mask if present
        if (causalMask != null) {
            // Mask shape check & alignment
            if (causalMask.getShape().get(0) == seqLen && causalMask.getShape().get(1) == keySeqLen) {
                NDArray mask3D = causalMask.expandDims(0);
                scores = scores.add(mask3D);
            }
        }

        // Softmax across key sequence dimension
        NDArray maxScores = scores.max(new int[]{2}, true);
        NDArray expScores = scores.sub(maxScores).exp();
        NDArray sumExp = expScores.sum(new int[]{2}, true);
        NDArray attnWeights = expScores.div(sumExp);

        // Context = AttnWeights @ V -> [numHeads, seqLen, headDim]
        NDArray context = attnWeights.matMul(V);

        // Transpose back to [seqLen, numHeads, headDim] -> reshape to [seqLen, dModel]
        NDArray concatContext = context.transpose(1, 0, 2).reshape(seqLen, dModel);

        // Final output projection
        NDArray output = concatContext.matMul(wO);

        return new AttentionResult(output, newKCache, newVCache);
    }
}