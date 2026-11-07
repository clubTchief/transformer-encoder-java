package com.example.math;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.List;

/**
 * One Pre-LN decoder layer: causal multi-head attention with RoPE + SwiGLU feed-forward.
 *
 * Unlike the inline code it replaces, every weight and LayerNorm is created ONCE in the constructor,
 * so the same parameters are reused (and trained) across forward passes. Build the stack before
 * opening the GradientCollector; only call forward() inside it.
 */
public class DecoderBlock {

    /** Intermediate tensors, kept so the pipeline's explain* printouts still work. */
    public record Trace(NDArray q, NDArray k, NDArray v, NDArray qSplit, NDArray attention,
                        NDArray mhaOutput, NDArray subLayer1Output,
                        NDArray ffnHidden, NDArray ffnOutput, NDArray output) {}

    private final int numHeads;
    private final int dModel;
    private final int dK;

    private final LayerNorm norm1;
    private final LayerNorm norm2;
    private final NDArray wQ, wK, wV, wO;
    private final NDArray wGate, bGate, w1, b1, w2, b2;

    public DecoderBlock(NDManager manager, int dModel, int numHeads, int dFF, float eps) {
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel must be divisible by numHeads");
        }
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dK = dModel / numHeads;

        norm1 = new LayerNorm(manager, dModel, eps);
        norm2 = new LayerNorm(manager, dModel, eps);

        float stdAttn = (float) Math.sqrt(2.0 / (dModel + dK));
        float stdO = (float) Math.sqrt(2.0 / (dModel + dModel));
        wQ = param(manager, stdAttn, new Shape(dModel, dModel));
        wK = param(manager, stdAttn, new Shape(dModel, dModel));
        wV = param(manager, stdAttn, new Shape(dModel, dModel));
        wO = param(manager, stdO, new Shape(dModel, dModel));

        float std1 = (float) Math.sqrt(2.0 / (dModel + dFF));
        float std2 = (float) Math.sqrt(2.0 / (dFF + dModel));
        wGate = param(manager, std1, new Shape(dModel, dFF));
        bGate = zeros(manager, new Shape(dFF));
        w1 = param(manager, std1, new Shape(dModel, dFF));
        b1 = zeros(manager, new Shape(dFF));
        w2 = param(manager, std2, new Shape(dFF, dModel));
        b2 = zeros(manager, new Shape(dModel));
    }

    /**
     * @param x        [seqLen, dModel]
     * @param mask     additive causal mask [seqLen, seqLen]
     * @param ropeCos  from ModernizedHierarchicalEncoderPipeline.buildRoPETables
     * @param ropeSin  from ModernizedHierarchicalEncoderPipeline.buildRoPETables
     */
    public Trace forward(NDArray x, NDArray mask, NDArray ropeCos, NDArray ropeSin) {
        long seqLen = x.getShape().get(0);

        // ---- SubLayer 1: Pre-LN + RoPE multi-head attention with causal mask ----
        NDArray n1 = norm1.forward(x);
        NDArray q = n1.matMul(wQ);
        NDArray k = n1.matMul(wK);
        NDArray v = n1.matMul(wV);

        NDArray qSplit = q.reshape(new Shape(seqLen, numHeads, dK)).transpose(1, 0, 2);
        NDArray kSplit = k.reshape(new Shape(seqLen, numHeads, dK)).transpose(1, 0, 2);
        NDArray vSplit = v.reshape(new Shape(seqLen, numHeads, dK)).transpose(1, 0, 2);

        NDArray qRope = ModernizedHierarchicalEncoderPipeline.applyRoPE(qSplit, ropeCos, ropeSin);
        NDArray kRope = ModernizedHierarchicalEncoderPipeline.applyRoPE(kSplit, ropeCos, ropeSin);

        NDArray scores = qRope.matMul(kRope.transpose(0, 2, 1)).div(Math.sqrt(dK)).add(mask);
        NDArray attention = scores.softmax(-1);
        NDArray heads = attention.matMul(vSplit);
        NDArray mhaOutput = heads.transpose(1, 0, 2).reshape(new Shape(seqLen, dModel)).matMul(wO);
        NDArray subLayer1Output = x.add(mhaOutput);

        // ---- SubLayer 2: Pre-LN + SwiGLU feed-forward ----
        NDArray n2 = norm2.forward(subLayer1Output);
        NDArray gate = ModernizedHierarchicalEncoderPipeline.applySwish(n2.matMul(wGate).add(bGate));
        NDArray ffnHidden = gate.mul(n2.matMul(w1).add(b1));
        NDArray ffnOutput = ffnHidden.matMul(w2).add(b2);
        NDArray output = subLayer1Output.add(ffnOutput);

        return new Trace(q, k, v, qSplit, attention, mhaOutput, subLayer1Output, ffnHidden, ffnOutput, output);
    }

    public NDArray apply(NDArray x, NDArray mask, NDArray ropeCos, NDArray ropeSin) {
        return forward(x, mask, ropeCos, ropeSin).output();
    }

    /** All trainable tensors of this block, in a fixed order. */
    public List<NDArray> params() {
        List<NDArray> p = new ArrayList<>();
        p.add(norm1.getGamma());
        p.add(norm1.getBeta());
        p.add(wQ);
        p.add(wK);
        p.add(wV);
        p.add(wO);
        p.add(norm2.getGamma());
        p.add(norm2.getBeta());
        p.add(wGate);
        p.add(bGate);
        p.add(w1);
        p.add(b1);
        p.add(w2);
        p.add(b2);
        return p;
    }

    private static NDArray param(NDManager manager, float std, Shape shape) {
        NDArray a = manager.randomNormal(0.0f, std, shape, DataType.FLOAT32);
        a.setRequiresGradient(true);
        return a;
    }

    private static NDArray zeros(NDManager manager, Shape shape) {
        NDArray a = manager.zeros(shape, DataType.FLOAT32);
        a.setRequiresGradient(true);
        return a;
    }
}
