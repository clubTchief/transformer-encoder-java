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
     * Creates DJL GradientCollector context for autograd tracking.
     */
    public GradientCollector createCollector() {
        return Engine.getInstance().newGradientCollector();
    }

    /**
     * Executes SGD parameter update step across registered model parameters.
     */
    public void step(List<NDArray> parameters) {
        for (NDArray param : parameters) {
            if (param.hasGradient()) {
                NDArray grad = param.getGradient();
                param.subi(grad.mul(learningRate));
            }
        }
    }
}