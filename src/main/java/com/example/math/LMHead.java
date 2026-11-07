package com.example.math;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.List;

public class LMHead {

    private final NDArray W_vocab; // Shape: [dModel, vocabSize]
    private final int vocabSize;

    public LMHead(NDManager manager, int dModel, int vocabSize) {
        this.vocabSize = vocabSize;
        float stddev = (float) Math.sqrt(2.0 / (dModel + vocabSize));
        this.W_vocab = manager.randomNormal(0.0f, stddev, new Shape(dModel, vocabSize), DataType.FLOAT32);
        this.W_vocab.setRequiresGradient(true);
    }

    /**
     * Projects transformer output hidden states to vocabulary logits.
     * Output shape: [seqLen, vocabSize]
     */
    public NDArray forward(NDArray hiddenStates) {
        return hiddenStates.matMul(W_vocab);
    }

    public List<NDArray> params() {
        return List.of(W_vocab);
    }
}