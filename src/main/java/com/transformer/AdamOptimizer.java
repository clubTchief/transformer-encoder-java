package com.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

import java.util.ArrayList;
import java.util.List;

public class AdamOptimizer {

    private final List<NDArray> params;
    private final float lr;
    private final float beta1;
    private final float beta2;
    private final float eps;

    private final List<NDArray> m;
    private final List<NDArray> v;
    private int t;

    public AdamOptimizer(List<NDArray> params, float lr, float beta1, float beta2, float eps) {
        this.params = params;
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.t = 0;

        this.m = new ArrayList<>();
        this.v = new ArrayList<>();

        // Initialize first and second moment trackers with zeros matching param shapes
        for (NDArray param : params) {
            m.add(param.getManager().zeros(param.getShape(), param.getDataType()));
            v.add(param.getManager().zeros(param.getShape(), param.getDataType()));
        }
    }

    public AdamOptimizer(List<NDArray> params, float lr) {
        this(params, lr, 0.9f, 0.999f, 1e-8f);
    }

    public void step() {
        t++;

        // Compute bias-correction scaling factors
        float biasCorrection1 = 1.0f - (float) Math.pow(beta1, t);
        float biasCorrection2 = 1.0f - (float) Math.pow(beta2, t);

        for (int i = 0; i < params.size(); i++) {
            NDArray param = params.get(i);
            NDArray grad = param.getGradient();

            if (grad == null) {
                continue;
            }

            NDArray mTrack = m.get(i);
            NDArray vTrack = v.get(i);

            // 1. Update biased 1st moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * grad
            mTrack.muli(beta1).addi(grad.mul(1.0f - beta1));

            // 2. Update biased 2nd raw moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * (grad^2)
            vTrack.muli(beta2).addi(grad.square().mul(1.0f - beta2));

            // 3. Compute bias-corrected estimates: m_hat = m_t / (1 - beta1^t), v_hat = v_t / (1 - beta2^t)
            NDArray mHat = mTrack.div(biasCorrection1);
            NDArray vHat = vTrack.div(biasCorrection2);

            // 4. Compute parameter update step: delta = lr * m_hat / (sqrt(v_hat) + eps)
            NDArray denom = vHat.sqrt().add(eps);
            NDArray step = mHat.div(denom).mul(lr);

            // 5. In-place gradient descent update: param = param - step
            param.subi(step);
        }
    }

    public int getStepCount() {
        return t;
    }
}