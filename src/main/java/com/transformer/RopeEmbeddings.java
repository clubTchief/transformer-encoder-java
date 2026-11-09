package com.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;

/**
 * The ONE rotary position embedding implementation in this code base (rotate-half pairing: channel i of a head
 * is rotated together with channel i + headDim/2).
 * <p>
 * The tables and {@link #applyRope} belong together: the tables repeat the frequencies as two identical halves
 * [f0 .. f(h-1), f0 .. f(h-1)], which is exactly what rotate-half expects. Do not mix them with tables built for
 * the interleaved (2i, 2i+1) convention; that is not a rotation any more and silently breaks the property that
 * attention scores depend only on the relative distance between tokens.
 */
public final class RopeEmbeddings {

    private RopeEmbeddings() {
    }

    /**
     * Precomputes RoPE cosine and sine tables of shape [maxSeqLen, headDim]; row p is the rotation for ABSOLUTE
     * position p.
     */
    public static NDArray[] computeCosSin(NDManager manager, int maxSeqLen, int headDim, float theta) {
        if (headDim % 2 != 0) {
            throw new IllegalArgumentException("RoPE needs an even head dimension but got " + headDim);
        }
        NDArray dimIndices = manager.arange(0, headDim, 2, DataType.FLOAT32);
        NDArray invFreq = manager.create(1.0f).div(
                manager.create(theta).pow(dimIndices.div(headDim))
        );

        NDArray position = manager.arange(0, maxSeqLen, 1, DataType.FLOAT32);
        NDArray freqs = position.expandDims(1).matMul(invFreq.expandDims(0));

        NDArray emb = freqs.concat(freqs, 1);
        return new NDArray[]{emb.cos(), emb.sin()};
    }

    public static NDArray[] computeCosSin(NDManager manager, int maxSeqLen, int headDim) {
        return computeCosSin(manager, maxSeqLen, headDim, 10000.0f);
    }

    /**
     * Rows [start, start + length) of the tables: the rotations for the tokens at those absolute positions.
     * A token that is decoded after {@code start} cached tokens needs row {@code start}.
     */
    public static NDArray[] slice(NDArray cos, NDArray sin, int start, int length) {
        int rank = cos.getShape().dimension();
        long total = cos.getShape().get(rank - 2);
        if (start < 0 || length < 1 || start + length > total) {
            throw new IllegalArgumentException("Cannot take positions [" + start + ", " + (start + length)
                    + ") from RoPE tables that only cover " + total + " positions");
        }
        String pattern = ":, ".repeat(rank - 2) + start + ":" + (start + length) + ", :";
        return new NDArray[]{cos.get(pattern), sin.get(pattern)};
    }

    /**
     * Applies Rotary Position Embedding to Query or Key tensors of shape [batch, numHeads, seqLen, headDim].
     * cos/sin must end with [seqLen, headDim] (2-D tables are broadcast over batch and heads).
     */
    public static NDArray applyRope(NDArray x, NDArray cos, NDArray sin) {
        if (x.getShape().dimension() != 4) {
            throw new IllegalArgumentException("applyRope expects [batch, numHeads, seqLen, headDim] but got " + x.getShape());
        }
        long headDim = x.getShape().get(3);
        long halfDim = headDim / 2;

        NDArray x1 = x.get("..., :" + halfDim);
        NDArray x2 = x.get("..., " + halfDim + ":");

        // Rotate half: [-x2, x1]
        NDArray rotatedX = x2.neg().concat(x1, -1);

        NDArray cosExpanded = cos;
        NDArray sinExpanded = sin;
        if (cos.getShape().dimension() == 2) {
            cosExpanded = cos.expandDims(0).expandDims(0);
            sinExpanded = sin.expandDims(0).expandDims(0);
        }

        return x.mul(cosExpanded).add(rotatedX.mul(sinExpanded));
    }
}
