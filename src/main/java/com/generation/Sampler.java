package com.generation;

import ai.djl.ndarray.NDArray;
import java.util.Random;

public class Sampler {

    private final Random random;

    public Sampler() {
        this.random = new Random();
    }

    public Sampler(long seed) {
        this.random = new Random(seed);
    }

    /**
     * Greedy sampling: returns token ID with highest probability.
     */
    public int sampleGreedy(NDArray logits) {
        return (int) logits.argMax(-1).getLong();
    }

    /**
     * Categorical/Multinomial sampling from probability distribution.
     */
    public int sampleMultinomial(NDArray logits) {
        NDArray probs = logits.softmax(-1);
        float[] probArray = probs.toFloatArray();

        float r = random.nextFloat();
        float cumulativeProb = 0.0f;

        for (int i = 0; i < probArray.length; i++) {
            cumulativeProb += probArray[i];
            if (r <= cumulativeProb) {
                return i;
            }
        }

        return probArray.length - 1;
    }
}