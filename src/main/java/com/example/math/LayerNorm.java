package com.example.math;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class LayerNorm {

    private final NDArray gamma;
    private final NDArray beta;
    private final float epsilon;

    public LayerNorm(NDManager manager, int dModel, float epsilon) {
        this.epsilon = epsilon;
        
        // Persistent learnable parameters with autograd enabled
        this.gamma = manager.ones(new Shape(dModel), DataType.FLOAT32);
        this.gamma.setRequiresGradient(true);

        this.beta = manager.zeros(new Shape(dModel), DataType.FLOAT32);
        this.beta.setRequiresGradient(true);
    }

    public NDArray forward(NDArray X) {
        NDArray mean = X.mean(new int[]{-1}, true);
        NDArray centered = X.sub(mean);
        NDArray variance = centered.square().mean(new int[]{-1}, true);

        NDArray stdDev = variance.add(epsilon).sqrt();
        NDArray normalized = centered.div(stdDev);

        return normalized.mul(gamma).add(beta);
    }

    public NDArray getGamma() {
        return gamma;
    }

    public NDArray getBeta() {
        return beta;
    }
}