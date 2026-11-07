package com.example.math;

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
}