package com.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

import java.util.ArrayList;
import java.util.List;

/**
 * Adam / AdamW. {@link #step(float, float, float)} adds what real training needs on top of plain Adam: a learning
 * rate per step (warm-up and decay schedules), decoupled weight decay for matrices and global gradient-norm clipping.
 * All temporaries of a step live in a short-lived sub-manager, so thousands of steps do not pile up native memory.
 * Call step() OUTSIDE the GradientCollector block. It also resets the gradients, because backward() ADDS to them.
 */
public class AdamOptimizer {

    private final List<NDArray> params;
    private final float lr;
    private final float beta1;
    private final float beta2;
    private final float eps;

    private final List<NDArray> m;
    private final List<NDArray> v;
    private int t;
    private float lastGradNorm;

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

    /** One plain Adam step with the learning rate given at construction (no decay, no clipping). */
    public void step() {
        step(lr, 0f, 0f);
    }

    /**
     * @param learningRate learning rate for THIS step
     * @param weightDecay  decoupled weight decay (AdamW), applied to matrices only; 0 disables
     * @param maxGradNorm  clip the global gradient norm to this value; 0 or less disables
     */
    public void step(float learningRate, float weightDecay, float maxGradNorm) {
        if (params.isEmpty()) {
            return;
        }
        t++;
        float biasCorrection1 = 1.0f - (float) Math.pow(beta1, t);
        float biasCorrection2 = 1.0f - (float) Math.pow(beta2, t);

        try (NDManager tmp = params.get(0).getManager().newSubManager()) {
            int n = params.size();
            NDArray[] grads = new NDArray[n];
            double sumSquares = 0;
            for (int i = 0; i < n; i++) {
                NDArray grad = params.get(i).getGradient();
                if (grad == null) {
                    continue;
                }
                grad.attach(tmp);
                grads[i] = grad;
                if (maxGradNorm > 0f) {
                    sumSquares += grad.square().sum().getFloat();
                }
            }

            float clipScale = 1f;
            if (maxGradNorm > 0f) {
                double norm = Math.sqrt(sumSquares);
                lastGradNorm = (float) norm;
                if (norm > maxGradNorm) {
                    clipScale = (float) (maxGradNorm / (norm + 1e-6));
                }
            }

            for (int i = 0; i < n; i++) {
                if (grads[i] == null) {
                    continue;
                }
                NDArray param = params.get(i);
                NDArray grad = clipScale == 1f ? grads[i] : grads[i].mul(clipScale);
                NDArray mTrack = m.get(i);
                NDArray vTrack = v.get(i);

                // m_t = beta1 * m_{t-1} + (1 - beta1) * g        v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
                mTrack.muli(beta1).addi(grad.mul(1.0f - beta1));
                vTrack.muli(beta2).addi(grad.square().mul(1.0f - beta2));

                // bias-corrected estimates; created on the moment's manager, so hand them to the step's manager
                NDArray mHat = mTrack.div(biasCorrection1);
                mHat.attach(tmp);
                NDArray vHat = vTrack.div(biasCorrection2);
                vHat.attach(tmp);
                NDArray update = mHat.div(vHat.sqrt().add(eps)).mul(learningRate);

                if (weightDecay > 0f && param.getShape().dimension() == 2) {
                    param.muli(1f - learningRate * weightDecay);
                }
                param.subi(update);

                // backward() ADDS to the existing gradient: reset it or every step would also contain all earlier ones
                grads[i].muli(0f);
            }
        }
    }

    /** Global gradient norm measured (before clipping) in the last step() that had clipping enabled. */
    public float getLastGradNorm() {
        return lastGradNorm;
    }

    public int getStepCount() {
        return t;
    }
}
