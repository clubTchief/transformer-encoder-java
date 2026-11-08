package com.decoder;

import ai.djl.ndarray.NDArray;

public class DecoderBlockResult {

    public final NDArray output;
    public final NDArray kCache;
    public final NDArray vCache;

    public DecoderBlockResult(NDArray output, NDArray kCache, NDArray vCache) {
        this.output = output;
        this.kCache = kCache;
        this.vCache = vCache;
    }

    public DecoderBlockResult(NDArray output) {
        this(output, null, null);
    }

    public NDArray getOutput() {
        return output;
    }

    public NDArray getCachedKey() {
        return kCache;
    }

    public NDArray getCachedValue() {
        return vCache;
    }

    public boolean hasCache() {
        return kCache != null && vCache != null;
    }
}