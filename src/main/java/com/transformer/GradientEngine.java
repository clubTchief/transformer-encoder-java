package com.transformer;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.training.GradientCollector;

import java.util.List;

public class GradientEngine {

    private final float learningRate;

    public GradientEngine(float learningRate) {
        this.learningRate = learningRate;
    }

    public GradientCollector createCollector() {
        return Engine.getInstance().newGradientCollector();
    }

    /**
     * Updates model parameters via standard SGD and clears/zeroes accumulated gradients safely.
     */
    public void step(List<NDArray> parameters) {
        for (NDArray param : parameters) {
            if (param.hasGradient()) {
                NDArray grad = param.getGradient();
                if (grad != null) {
                    // 1. Calculate SGD step and update parameter storage safely
                    try (NDArray stepSize = grad.mul(learningRate);
                         NDArray updated = param.sub(stepSize)) {
                        param.set(updated.toFloatArray());
                    }

                    // 2. Safely zero out the gradient array without in-place PyTorch autograd mutation
                    float[] zeroArray = new float[(int) grad.size()];
                    grad.set(zeroArray);
                }
            }
        }
    }

    /**
     * Optional explicit helper to zero out gradients across parameter lists.
     */
    public void zeroGradients(List<NDArray> parameters) {
        for (NDArray param : parameters) {
            if (param.hasGradient()) {
                NDArray grad = param.getGradient();
                if (grad != null) {
                    float[] zeroArray = new float[(int) grad.size()];
                    grad.set(zeroArray);
                }
            }
        }
    }
}