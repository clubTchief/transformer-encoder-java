package com.transformer;

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
        this.gamma = manager.ones(new Shape(dModel), DataType.FLOAT32);
        this.gamma.setRequiresGradient(true);
    }

    public NDArray forward(NDArray x) {
        // Calculate Root Mean Square along the last dimension (-1)
        NDArray variance = x.square().mean(new int[]{-1}, true);
        NDArray rms = variance.add(eps).sqrt();
        NDArray normX = x.div(rms);

        // Broadcast gamma across input dimensions if 3D [batch, seqLen, dModel]
        if (x.getShape().dimension() == 3) {
            NDArray gamma3D = gamma.expandDims(0).expandDims(0); // Shape [1, 1, dModel]
            return normX.mul(gamma3D);
        }

        return normX.mul(gamma);
    }

    public List<NDArray> params() {
        return Collections.singletonList(gamma);
    }
}