package com.decoder;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.transformer.CausalMaskGenerator;
import com.transformer.RMSNorm;

import java.util.ArrayList;
import java.util.List;

public class DecoderBlock {

    private final int dModel;
    private final int numHeads;
    private final int headDim;
    private final int dFF;
    private final float eps;

    // Attention Weight & Bias parameters (7 total)
    private final NDArray wQ, bQ;
    private final NDArray wK, bK;
    private final NDArray wV, bV;
    private final NDArray wO;

    // Layer Normalization parameters (4 total)
    private final RMSNorm norm1;
    private final RMSNorm norm2;

    // SwiGLU Feed-Forward parameters (3 total)
    private final NDArray wG;
    private final NDArray wU;
    private final NDArray wD;

    public DecoderBlock(NDManager manager, int dModel, int numHeads, int dFF, float eps) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.headDim = dModel / numHeads;
        this.dFF = dFF;
        this.eps = eps;

        float scale = (float) Math.sqrt(2.0 / dModel);

        // Attention Weights & Biases
        this.wQ = manager.randomUniform(-scale, scale, new Shape(dModel, dModel), DataType.FLOAT32);
        this.wK = manager.randomUniform(-scale, scale, new Shape(dModel, dModel), DataType.FLOAT32);
        this.wV = manager.randomUniform(-scale, scale, new Shape(dModel, dModel), DataType.FLOAT32);
        this.wO = manager.randomUniform(-scale, scale, new Shape(dModel, dModel), DataType.FLOAT32);

        this.bQ = manager.zeros(new Shape(dModel), DataType.FLOAT32);
        this.bK = manager.zeros(new Shape(dModel), DataType.FLOAT32);
        this.bV = manager.zeros(new Shape(dModel), DataType.FLOAT32);

        // Layer Normalization
        this.norm1 = new RMSNorm(manager, dModel, eps);
        this.norm2 = new RMSNorm(manager, dModel, eps);

        // SwiGLU FFN Weights
        float ffnScale = (float) Math.sqrt(2.0 / dModel);
        this.wG = manager.randomUniform(-ffnScale, ffnScale, new Shape(dModel, dFF), DataType.FLOAT32);
        this.wU = manager.randomUniform(-ffnScale, ffnScale, new Shape(dModel, dFF), DataType.FLOAT32);
        this.wD = manager.randomUniform(-ffnScale, ffnScale, new Shape(dFF, dModel), DataType.FLOAT32);

        // Explicitly enable gradients on all 14 parameters
        for (NDArray p : params()) {
            p.setRequiresGradient(true);
        }
    }

    public NDArray forward(NDArray inputX, NDArray causalMask, NDArray arg3, NDArray arg4) {
        return forward(inputX.getManager(), inputX, causalMask);
    }

    public NDArray forward(NDManager manager, NDArray inputX, NDArray causalMask) {
        NDArray normX1 = norm1.forward(inputX);
        NDArray attnOut = selfAttention(manager, normX1, causalMask);
        NDArray x1 = inputX.add(attnOut);

        NDArray normX2 = norm2.forward(x1);
        NDArray ffnOut = swiGLU(manager, normX2);
        return x1.add(ffnOut);
    }

    /**
     * Original API order: (x, mask, cos, sin, kCache, vCache). cos/sin are accepted for compatibility but are
     * currently unused because this block applies no RoPE. (This overload used to read its 3rd/4th arguments
     * as the caches and drop the real ones, so callers silently got a bogus "past" built from cos/sin.)
     */
    public DecoderBlockResult forwardWithCache(NDArray inputX, NDArray causalMask, NDArray cos, NDArray sin,
                                               NDArray pastK, NDArray pastV) {
        return forwardWithCache(inputX.getManager(), inputX, causalMask, pastK, pastV);
    }

    public DecoderBlockResult forwardWithCache(NDArray inputX, NDArray causalMask, NDArray pastK, NDArray pastV) {
        return forwardWithCache(inputX.getManager(), inputX, causalMask, pastK, pastV);
    }

public DecoderBlockResult forwardWithCache(NDManager manager, NDArray inputX, NDArray causalMask, NDArray pastK, NDArray pastV) {
    boolean is2D = inputX.getShape().dimension() == 2;
    NDArray x3d = is2D ? inputX.expandDims(0) : inputX;

    NDArray normX1 = norm1.forward(x3d);
    
    Shape shape = normX1.getShape();
    long batchSize = shape.get(0);
    long seqLen = shape.get(1); // 1 during step-by-step decoding

    // 1. Linear Projections for current step input
    NDArray q = normX1.matMul(wQ).add(bQ);
    NDArray k = normX1.matMul(wK).add(bK);
    NDArray v = normX1.matMul(wV).add(bV);

    // Form 4D multi-head tensors: [batchSize, numHeads, seqLen, headDim]
    NDArray qHead = q.reshape(new Shape(batchSize, seqLen, numHeads, headDim)).transpose(0, 2, 1, 3);
    NDArray kHead = k.reshape(new Shape(batchSize, seqLen, numHeads, headDim)).transpose(0, 2, 1, 3);
    NDArray vHead = v.reshape(new Shape(batchSize, seqLen, numHeads, headDim)).transpose(0, 2, 1, 3);

    // 2. Append the new keys/values to the cache. Caches must be [batch, numHeads, pastLen, headDim];
    //    anything else is a caller bug and is rejected instead of being reshaped into something plausible.
    if ((pastK == null) != (pastV == null)) {
        throw new IllegalArgumentException("pastK and pastV must either both be null or both be provided");
    }
    NDArray currentK = appendToCache(pastK, kHead, "K", batchSize);
    NDArray currentV = appendToCache(pastV, vHead, "V", batchSize);

    // 4. Scaled Dot-Product Attention over dynamic sequence length
    NDArray scores = qHead.matMul(currentK.transpose(0, 1, 3, 2)).div(Math.sqrt(headDim));
    long totalSeqLen = currentK.getShape().get(2);

    // 5. Causal mask: must be exactly [seqLen, totalSeqLen] (null is allowed for a single new token).
    //    Use CausalMaskGenerator.createCachedMask(manager, seqLen, totalSeqLen). A wrong shape throws.
    CausalMaskGenerator.requireAttentionMask(causalMask, seqLen, totalSeqLen);
    if (causalMask != null) {
        scores = scores.add(causalMask);
    }

    NDArray attnWeights = scores.softmax(-1);
    NDArray context = attnWeights.matMul(currentV);

    NDArray concat = context.transpose(0, 2, 1, 3).reshape(new Shape(batchSize, seqLen, dModel));
    NDArray attnOut = concat.matMul(wO);
    NDArray x1 = x3d.add(attnOut);

    // 6. Pre-LN SwiGLU FFN
    NDArray normX2 = norm2.forward(x1);
    NDArray ffnOut = swiGLU(manager, normX2);
    NDArray out3d = x1.add(ffnOut);

    NDArray finalOut = is2D ? out3d.squeeze(0) : out3d;

    return new DecoderBlockResult(finalOut, currentK, currentV);
}

    private NDArray appendToCache(NDArray past, NDArray fresh, String name, long batchSize) {
        if (past == null) {
            return fresh;
        }
        Shape s = past.getShape();
        if (s.dimension() != 4 || s.get(0) != batchSize || s.get(1) != numHeads || s.get(3) != headDim) {
            throw new IllegalArgumentException(name + " cache must have shape [" + batchSize + ", " + numHeads
                    + ", pastLen, " + headDim + "] but was " + s
                    + ". Pass the kCache/vCache returned by the previous forwardWithCache call (not cos/sin).");
        }
        return past.concat(fresh, 2);
    }

    private NDArray selfAttention(NDManager manager, NDArray x, NDArray causalMask) {
        Shape shape = x.getShape();
        boolean is3D = shape.dimension() == 3;
        
        long batchSize = is3D ? shape.get(0) : 1;
        long seqLen = is3D ? shape.get(1) : shape.get(0);

        NDArray q = x.matMul(wQ).add(bQ);
        NDArray k = x.matMul(wK).add(bK);
        NDArray v = x.matMul(wV).add(bV);

        NDArray qHead = q.reshape(new Shape(batchSize, seqLen, numHeads, headDim)).transpose(0, 2, 1, 3);
        NDArray kHead = k.reshape(new Shape(batchSize, seqLen, numHeads, headDim)).transpose(0, 2, 1, 3);
        NDArray vHead = v.reshape(new Shape(batchSize, seqLen, numHeads, headDim)).transpose(0, 2, 1, 3);

        NDArray scores = qHead.matMul(kHead.transpose(0, 1, 3, 2)).div(Math.sqrt(headDim));

        CausalMaskGenerator.requireAttentionMask(causalMask, seqLen, seqLen);
        if (causalMask != null) {
            scores = scores.add(causalMask);
        }

        NDArray attnWeights = scores.softmax(-1);
        NDArray context = attnWeights.matMul(vHead);

        NDArray concat = context.transpose(0, 2, 1, 3).reshape(is3D ? new Shape(batchSize, seqLen, dModel) : new Shape(seqLen, dModel));
        return concat.matMul(wO);
    }

    private NDArray swiGLU(NDManager manager, NDArray x) {
        NDArray gate = x.matMul(wG);
        NDArray up = x.matMul(wU);

        // Differentiable, element-wise Sigmoid: 1 / (1 + exp(-gate))
        NDArray sigmoidGate = gate.neg().exp().add(1.0f).pow(-1.0f);
        NDArray swishGate = gate.mul(sigmoidGate);
        NDArray hidden = swishGate.mul(up);

        return hidden.matMul(wD);
    }

    public List<NDArray> params() {
        List<NDArray> p = new ArrayList<>();
        p.add(wQ); p.add(bQ);
        p.add(wK); p.add(bK);
        p.add(wV); p.add(bV);
        p.add(wO);
        p.addAll(norm1.params());
        p.addAll(norm2.params());
        p.add(wG);
        p.add(wU);
        p.add(wD);
        return p;
    }
}