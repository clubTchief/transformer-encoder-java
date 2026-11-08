package com.example.math;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.Collections;
import java.util.List;

public class RMSNorm {

    private final NDArray gamma;
    private final float eps;

    public RMSNorm(NDManager manager, int dModel, float eps) {
        this.eps = eps;
        // Initialize gain parameter gamma to ones (shape: [dModel])
        this.gamma = manager.ones(new Shape(dModel), DataType.FLOAT32);
    }

    public NDArray forward(NDArray x) {
        // x shape: [batchSize, seqLen, dModel] or [seqLen, dModel]
        NDManager manager = x.getManager();

        // 1. Mean of squares along the last dimension (dModel)
        NDArray meanSquare = x.square().mean(new int[]{-1}, true);

        // 2. RMS = sqrt(meanSquare + eps)
        NDArray rms = meanSquare.add(eps).sqrt();

        // 3. Normalize: x / RMS
        NDArray normalized = x.div(rms);

        // 4. Scale with learnable gain: normalized * gamma
        return normalized.mul(gamma);
    }

    public List<NDArray> params() {
        return Collections.singletonList(gamma);
    }

    public NDArray getGamma() {
        return gamma;
    }
}