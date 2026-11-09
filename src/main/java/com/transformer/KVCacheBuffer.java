package com.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

/**
 * Pre-allocated key/value cache for ONE attention layer. Inference only: it is filled in place, so it is not
 * part of the autograd graph. Keys must be stored AFTER RoPE has been applied to them.
 */
public class KVCacheBuffer {

    private final NDArray kBuffer;
    private final NDArray vBuffer;
    private final int batchSize;
    private final int numHeads;
    private final int maxSeqLen;
    private final int headDim;
    private int currentLength;

    public KVCacheBuffer(NDManager manager, int batchSize, int numHeads, int maxSeqLen, int headDim) {
        this.batchSize = batchSize;
        this.numHeads = numHeads;
        this.maxSeqLen = maxSeqLen;
        this.headDim = headDim;
        this.kBuffer = manager.zeros(new Shape(batchSize, numHeads, maxSeqLen, headDim), DataType.FLOAT32);
        this.vBuffer = manager.zeros(new Shape(batchSize, numHeads, maxSeqLen, headDim), DataType.FLOAT32);
        this.currentLength = 0;
    }

    /**
     * Appends new Key and Value projections in place.
     *
     * @param kFresh 4D tensor [batchSize, numHeads, stepSeqLen, headDim]
     * @param vFresh 4D tensor [batchSize, numHeads, stepSeqLen, headDim]
     */
    public void update(NDArray kFresh, NDArray vFresh) {
        Shape ks = kFresh.getShape();
        if (ks.dimension() != 4 || ks.get(0) != batchSize || ks.get(1) != numHeads || ks.get(3) != headDim) {
            throw new IllegalArgumentException("Cache expects [" + batchSize + ", " + numHeads + ", step, " + headDim
                    + "] but got " + ks);
        }
        if (!ks.equals(vFresh.getShape())) {
            throw new IllegalArgumentException("Key and value shapes differ: " + ks + " vs " + vFresh.getShape());
        }
        long stepSeqLen = ks.get(2);
        if (currentLength + stepSeqLen > maxSeqLen) {
            throw new IllegalStateException("KV cache overflow: capacity " + maxSeqLen + ", already holds "
                    + currentLength + ", cannot append " + stepSeqLen);
        }
        int startPos = this.currentLength;
        int endPos = startPos + (int) stepSeqLen;

        NDIndex index = new NDIndex(String.format(":, :, %d:%d, :", startPos, endPos));
        this.kBuffer.set(index, kFresh);
        this.vBuffer.set(index, vFresh);

        this.currentLength = endPos;
    }

    /** Forgets everything. Old contents stay in memory but are never read: they are overwritten before use. */
    public void reset() {
        this.currentLength = 0;
    }

    public NDArray getActiveKeys() {
        return kBuffer.get(":, :, 0:" + currentLength + ", :");
    }

    public NDArray getActiveValues() {
        return vBuffer.get(":, :, 0:" + currentLength + ", :");
    }

    public int getCurrentLength() {
        return currentLength;
    }

    public int getCapacity() {
        return maxSeqLen;
    }

    public NDArray getKBuffer() {
        return kBuffer;
    }

    public NDArray getVBuffer() {
        return vBuffer;
    }
}
