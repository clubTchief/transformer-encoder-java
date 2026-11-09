package com.transformer;

/**
 * Architecture of a {@link BlockStackLM}. A record instead of seven positional ints: the old constructor
 * took (numLayers, dModel, numHeads, ...) in one place and (dModel, numHeads, numLayers, ...) in others, and
 * since they are all ints the compiler could not tell the difference.
 */
public record LMConfig(int numLayers, int dModel, int numHeads, int dFF, int vocabSize, float eps) {

    public LMConfig {
        if (numLayers < 1 || dModel < 1 || numHeads < 1 || dFF < 1 || vocabSize < 1) {
            throw new IllegalArgumentException("All sizes must be positive: " + this);
        }
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel (" + dModel + ") must be divisible by numHeads (" + numHeads + ")");
        }
        if ((dModel / numHeads) % 2 != 0) {
            throw new IllegalArgumentException("RoPE needs an even head dimension but dModel/numHeads = " + (dModel / numHeads));
        }
    }

    public int headDim() {
        return dModel / numHeads;
    }
}
