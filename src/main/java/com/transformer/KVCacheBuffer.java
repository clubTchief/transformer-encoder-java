package com.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class KVCacheBuffer {

    private final NDArray kBuffer;
    private final NDArray vBuffer;
    private int currentLength;

    public KVCacheBuffer(NDManager manager, int batchSize, int numHeads, int maxSeqLen, int headDim) {
        this.kBuffer = manager.zeros(new Shape(batchSize, numHeads, maxSeqLen, headDim), DataType.FLOAT32);
        this.vBuffer = manager.zeros(new Shape(batchSize, numHeads, maxSeqLen, headDim), DataType.FLOAT32);
        this.currentLength = 0;
    }

    /**
     * Updates the cache buffer in-place with new Key and Value projections.
     *
     * @param kFresh 4D Tensor [batchSize, numHeads, stepSeqLen, headDim]
     * @param vFresh 4D Tensor [batchSize, numHeads, stepSeqLen, headDim]
     */
    public void update(NDArray kFresh, NDArray vFresh) {
        long stepSeqLen = kFresh.getShape().get(2);
        int startPos = this.currentLength;
        int endPos = startPos + (int) stepSeqLen;

        // In-place slicing update: buffer[:, :, startPos:endPos, :] = fresh
        String slicePattern = String.format(":, :, %d:%d, :", startPos, endPos);
        ai.djl.ndarray.index.NDIndex index = new ai.djl.ndarray.index.NDIndex(slicePattern);

        this.kBuffer.set(index, kFresh);
        this.vBuffer.set(index, vFresh);

        this.currentLength = endPos;
    }

    /**
     * Slices valid active keys up to the current position for attention calculation.
     */
    public NDArray getActiveKeys() {
        return kBuffer.get(":, :, 0:" + currentLength + ", :");
    }

    /**
     * Slices valid active values up to the current position for attention calculation.
     */
    public NDArray getActiveValues() {
        return vBuffer.get(":, :, 0:" + currentLength + ", :");
    }

    public int getCurrentLength() {
        return currentLength;
    }

    public NDArray getKBuffer() {
        return kBuffer;
    }

    public NDArray getVBuffer() {
        return vBuffer;
    }
}