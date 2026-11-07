package com.example.story;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

import java.util.List;

/**
 * Minimal AdamW with global-norm gradient clipping, written on plain NDArray ops so it does not
 * depend on which optimizer kernels a given DJL engine implements. Call {@link #step} OUTSIDE the
 * GradientCollector block (grad mode off), otherwise in-place updates on leaf tensors are rejected.
 */
public final class AdamW {
    private static final float B1 = 0.9f;
    private static final float B2 = 0.95f;
    private static final float EPS = 1e-8f;

    private final List<NDArray> params;
    private final NDArray[] m;
    private final NDArray[] v;
    private final NDManager manager;
    private int t = 0;

    public AdamW(NDManager manager, List<NDArray> params) {
        this.manager = manager;
        this.params = params;
        this.m = new NDArray[params.size()];
        this.v = new NDArray[params.size()];
        for (int i = 0; i < params.size(); i++) {
            m[i] = manager.zeros(params.get(i).getShape());
            v[i] = manager.zeros(params.get(i).getShape());
        }
    }

    public void step(float lr, float weightDecay, float maxGradNorm) {
        t++;
        try (NDManager tmp = manager.newSubManager()) {
            int n = params.size();
            NDArray[] grads = new NDArray[n];
            double sq = 0;
            for (int i = 0; i < n; i++) {
                grads[i] = params.get(i).getGradient();
                grads[i].attach(tmp);
                sq += grads[i].square().sum().getFloat();
            }
            double norm = Math.sqrt(sq);
            float clip = norm > maxGradNorm ? (float) (maxGradNorm / (norm + 1e-6)) : 1f;
            float bc1 = 1f - (float) Math.pow(B1, t);
            float bc2 = 1f - (float) Math.pow(B2, t);

            for (int i = 0; i < n; i++) {
                NDArray p = params.get(i);
                NDArray g = grads[i].mul(clip);

                m[i].muli(B1).addi(g.mul(1f - B1));
                v[i].muli(B2).addi(g.square().mul(1f - B2));

                NDArray mHat = m[i].div(bc1);
                mHat.attach(tmp);
                NDArray vHat = v[i].div(bc2);
                vHat.attach(tmp);
                NDArray update = mHat.div(vHat.sqrt().add(EPS)).mul(lr);

                if (weightDecay > 0f && p.getShape().dimension() == 2) { // decay matrices only
                    p.muli(1f - lr * weightDecay);
                }
                p.subi(update);
                grads[i].muli(0f); // reset accumulated gradient for the next step
            }
        }
    }
}
