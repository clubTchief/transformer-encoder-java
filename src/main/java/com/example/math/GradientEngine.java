package com.example.math;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.training.GradientCollector;

import java.util.List;

public class GradientEngine {

    private final float learningRate;

    public GradientEngine(float learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Creates a GradientCollector for autograd tape tracking.
     */
    public GradientCollector createCollector() {
        return Engine.getInstance().newGradientCollector();
    }

    /**
     * Performs an SGD step across parameters.
     * Detaches gradient computation from the active graph to avoid
     * in-place leaf variable mutation and nested PtGradientCollector errors.
     */
    public void step(List<NDArray> parameters) {
        for (NDArray param : parameters) {
            if (param.hasGradient()) {
                NDArray grad = param.getGradient();
                if (grad != null) {
                    // 1. Calculate weighted update step tensor
                    // 2. Duplicate to break reference from autograd graph
                    // 3. Apply safe in-place subtraction
                    // Compute step tensor
                    try (NDArray stepSize = grad.mul(learningRate);
                         NDArray updated = param.sub(stepSize)) {
                        
                        // Replace internal tensor value cleanly without in-place PyTorch leaf mutation
                        param.set(updated.toFloatArray());
                    }
                    // Reset so the next backward() starts from zero instead of accumulating onto this step's gradient
                    grad.muli(0f);
                }
            }
        }
    }
}