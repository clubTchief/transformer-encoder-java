package com.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class CausalMaskGenerator {

    /**
     * Generates a square upper-triangular causal mask matrix.
     * 
     * @param manager  DJL NDManager instance
     * @param seqLen   Length of target sequence (t)
     * @return NDArray Mask matrix of shape [1, 1, seqLen, seqLen] filled with 0.0 for valid and -1e9 for masked
     */
    public static NDArray createCausalMask(NDManager manager, long seqLen) {
        // Create an upper triangular matrix (k=1 excludes main diagonal)
        NDArray ones = manager.ones(new Shape(seqLen, seqLen), DataType.FLOAT32);
        
        // Custom mask construction: upper-triangular elements set to -1e9f
        float[] maskValues = new float[(int) (seqLen * seqLen)];
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                if (j > i) {
                    maskValues[i * (int) seqLen + j] = -1e9f; // Mask out future tokens
                } else {
                    maskValues[i * (int) seqLen + j] = 0.0f;  // Keep past & present tokens
                }
            }
        }
        
        NDArray mask = manager.create(maskValues, new Shape(seqLen, seqLen));
        // Expand shape for Multi-Head Attention compatibility: [Batch=1, Heads=1, SeqLen, SeqLen]
        return mask.reshape(new Shape(1, 1, seqLen, seqLen));
    }

    /**
     * Mask for incremental decoding with a KV cache. The {@code newTokens} query rows are the LAST
     * positions of a sequence that now has {@code totalKeys} keys (cached + new).
     *
     * @return additive mask of shape [newTokens, totalKeys]: 0.0 where attention is allowed, -1e9 elsewhere
     */
    public static NDArray createCachedMask(NDManager manager, long newTokens, long totalKeys) {
        if (newTokens < 1 || totalKeys < newTokens) {
            throw new IllegalArgumentException("Need 1 <= newTokens <= totalKeys but got newTokens="
                    + newTokens + ", totalKeys=" + totalKeys);
        }
        float[] values = new float[(int) (newTokens * totalKeys)];
        long firstNewPosition = totalKeys - newTokens;
        for (int i = 0; i < newTokens; i++) {
            for (int j = 0; j < totalKeys; j++) {
                values[(int) (i * totalKeys + j)] = (j > firstNewPosition + i) ? -1e9f : 0.0f;
            }
        }
        return manager.create(values, new Shape(newTokens, totalKeys));
    }

    /**
     * Fails fast if an attention mask does not fit the score matrix it will be added to.
     * <p>
     * Contract: the mask is [queries, keys] (or [1, 1, queries, keys]), where {@code queries} is the number
     * of tokens being processed now and {@code keys} is the number of keys they attend to (cached + new).
     * A null mask is legal only for a single query token, which may attend to everything before it.
     * <p>
     * A mismatch used to be skipped silently, which turns a causal model into a bidirectional one
     * (future tokens leak in, training loss looks great, generation is garbage) without any error.
     */
    public static void requireAttentionMask(NDArray mask, long queries, long keys) {
        if (mask == null) {
            if (queries > 1) {
                throw new IllegalArgumentException("A causal mask is required when more than one token is processed ("
                        + queries + " query tokens, no mask). Use CausalMaskGenerator.createCausalMask(...) "
                        + "or createCachedMask(...).");
            }
            return;
        }
        Shape shape = mask.getShape();
        int rank = shape.dimension();
        if (rank != 2 && rank != 4) {
            throw new IllegalArgumentException("Attention mask must be 2-D [queries, keys] or 4-D [1, 1, queries, keys] but was "
                    + shape);
        }
        long maskQueries = shape.get(rank - 2);
        long maskKeys = shape.get(rank - 1);
        if (maskQueries != queries || maskKeys != keys) {
            throw new IllegalArgumentException("Attention mask " + shape + " does not match the attention scores: expected ["
                    + queries + ", " + keys + "] (queries x keys). For cached decoding use createCachedMask(manager, "
                    + queries + ", " + keys + ").");
        }
    }
}
