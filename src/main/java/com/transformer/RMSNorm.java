package com.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.List;

public class RMSNorm {

    private final NDArray gamma;
    private final NDArray beta;
    private final float eps;

    public RMSNorm(NDManager manager, int dModel, float eps) {
        this.eps = eps;
        this.gamma = manager.ones(new Shape(dModel), DataType.FLOAT32);
        this.beta = manager.zeros(new Shape(dModel), DataType.FLOAT32);

        // Enable autograd tracking on instantiation
        this.gamma.setRequiresGradient(true);
        this.beta.setRequiresGradient(true);
    }

    public NDArray forward(NDArray x) {
        NDArray rms = x.square().mean(new int[]{-1}, true).add(eps).sqrt();
        NDArray normX = x.div(rms);
        return normX.mul(gamma).add(beta);
    }

    public List<NDArray> params() {
        List<NDArray> p = new ArrayList<>();
        p.add(gamma);
        p.add(beta);
        return p;
    }
}