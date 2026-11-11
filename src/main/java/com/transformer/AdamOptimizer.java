package com.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

import java.util.ArrayList;
import java.util.List;

/**
 * Adam / AdamW. {@link #step(float, float, float)} adds what real training needs on top of plain Adam: a learning
 * rate per step (warm-up and decay schedules), decoupled weight decay for matrices, global gradient-norm clipping and
 * protection against non-finite gradients.
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
    private int skippedSteps;

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
     * @param maxGradNorm  clip the global gradient norm to this value; 0 or less disables clipping
     * @return false if the gradient contained NaN or Infinity: the step was SKIPPED (weights untouched, gradients
     *         cleared), because applying it would turn every weight into NaN. True if the update was applied.
     */
    public boolean step(float learningRate, float weightDecay, float maxGradNorm) {
        if (params.isEmpty()) {
            return true;
        }

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
                sumSquares += grad.square().sum().getFloat();
            }
            double norm = Math.sqrt(sumSquares);
            lastGradNorm = (float) norm;

            if (!Double.isFinite(norm)) {
                // NaN * 0 is still NaN, so the gradients have to be overwritten, not multiplied by zero
                for (NDArray grad : grads) {
                    if (grad != null) {
                        grad.set(new float[(int) grad.size()]);
                    }
                }
                skippedSteps++;
                return false;
            }

            t++;
            float biasCorrection1 = 1.0f - (float) Math.pow(beta1, t);
            float biasCorrection2 = 1.0f - (float) Math.pow(beta2, t);
            float clipScale = (maxGradNorm > 0f && norm > maxGradNorm) ? (float) (maxGradNorm / (norm + 1e-6)) : 1f;

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
        return true;
    }

    /** Global gradient norm of the last step(), measured before clipping. */
    public float getLastGradNorm() {
        return lastGradNorm;
    }

    /** Number of updates actually applied (skipped steps do not count). */
    public int getStepCount() {
        return t;
    }

    /** Steps that were skipped because the gradient was NaN or Infinity. */
    public int getSkippedSteps() {
        return skippedSteps;
    }
}
