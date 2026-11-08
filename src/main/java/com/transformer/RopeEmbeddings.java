package com.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class RopeEmbeddings {

    /**
     * Precomputes RoPE cosine and sine tables.
     */
    public static NDArray[] computeCosSin(NDManager manager, int maxSeqLen, int headDim, float theta) {
        NDArray dimIndices = manager.arange(0, headDim, 2, DataType.FLOAT32);
        NDArray invFreq = manager.create(1.0f).div(
            manager.create(theta).pow(dimIndices.div(headDim))
        );

        NDArray position = manager.arange(0, maxSeqLen, 1, DataType.FLOAT32);
        NDArray freqs = position.expandDims(1).matMul(invFreq.expandDims(0));

        NDArray emb = freqs.concat(freqs, 1);
        NDArray cos = emb.cos();
        NDArray sin = emb.sin();

        return new NDArray[]{cos, sin};
    }

    public static NDArray[] computeCosSin(NDManager manager, int maxSeqLen, int headDim) {
        return computeCosSin(manager, maxSeqLen, headDim, 10000.0f);
    }

    /**
     * Applies Rotary Position Embedding to Query or Key tensors.
     * Tensor shape expected: [batch, numHeads, seqLen, headDim]
     */
    public static NDArray applyRope(NDArray x, NDArray cos, NDArray sin) {
        long headDim = x.getShape().get(3);
        long halfDim = headDim / 2;

        // Split head dimension into x1 and x2
        NDArray x1 = x.get("..., :" + halfDim);
        NDArray x2 = x.get("..., " + halfDim + ":");

        // Rotate half: [-x2, x1]
        NDArray rotatedX = x2.neg().concat(x1, -1);

        // Expand cos and sin to broadcast over batch and numHeads dimensions: [1, 1, seqLen, headDim]
        NDArray cosExpanded = cos;
        NDArray sinExpanded = sin;
        if (cos.getShape().dimension() == 2) {
            cosExpanded = cos.expandDims(0).expandDims(0);
            sinExpanded = sin.expandDims(0).expandDims(0);
        }

        // x * cos + rotatedX * sin
        return x.mul(cosExpanded).add(rotatedX.mul(sinExpanded));
    }
}