package com.decoder;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.transformer.CausalMaskGenerator;
import com.transformer.KVCacheBuffer;
import com.transformer.RMSNorm;
import com.transformer.RopeEmbeddings;

import java.util.ArrayList;
import java.util.List;

public class DecoderBlock {

    private final int dModel;
    private final int numHeads;
    private final int headDim;
    private final int dFF;
    private final float eps;

    private final NDArray wQ, bQ;
    private final NDArray wK, bK;
    private final NDArray wV, bV;
    private final NDArray wO, bO;

    private final RMSNorm norm1;
    private final RMSNorm norm2;

    private final NDArray wG, bG;
    private final NDArray wU;
    private final NDArray wD, bD;

    public DecoderBlock(NDManager manager, int dModel, int numHeads, int dFF, float eps) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.headDim = dModel / numHeads;
        this.dFF = dFF;
        this.eps = eps;

        float scale = (float) Math.sqrt(2.0 / dModel);

        this.wQ = manager.randomUniform(-scale, scale, new Shape(dModel, dModel), DataType.FLOAT32);
        this.wK = manager.randomUniform(-scale, scale, new Shape(dModel, dModel), DataType.FLOAT32);
        this.wV = manager.randomUniform(-scale, scale, new Shape(dModel, dModel), DataType.FLOAT32);
        this.wO = manager.randomUniform(-scale, scale, new Shape(dModel, dModel), DataType.FLOAT32);

        this.bQ = manager.zeros(new Shape(dModel), DataType.FLOAT32);
        this.bK = manager.zeros(new Shape(dModel), DataType.FLOAT32);
        this.bV = manager.zeros(new Shape(dModel), DataType.FLOAT32);
        this.bO = manager.zeros(new Shape(dModel), DataType.FLOAT32);

        this.norm1 = new RMSNorm(manager, dModel, eps);
        this.norm2 = new RMSNorm(manager, dModel, eps);

        float ffnScale = (float) Math.sqrt(2.0 / dModel);
        this.wG = manager.randomUniform(-ffnScale, ffnScale, new Shape(dModel, dFF), DataType.FLOAT32);
        this.wU = manager.randomUniform(-ffnScale, ffnScale, new Shape(dModel, dFF), DataType.FLOAT32);
        this.wD = manager.randomUniform(-ffnScale, ffnScale, new Shape(dFF, dModel), DataType.FLOAT32);
        this.bG = manager.zeros(new Shape(dFF), DataType.FLOAT32);
        this.bD = manager.zeros(new Shape(dModel), DataType.FLOAT32);
        for (NDArray p : params()) {
            p.setRequiresGradient(true);
        }
    }

    /**
     * Efficient forward pass using pre-allocated static KV-Cache buffer.
     */
    public NDArray forwardWithBuffer(NDManager manager, NDArray inputX, NDArray causalMask,
                                     NDArray cos, NDArray sin, KVCacheBuffer cacheBuffer) {
        boolean is2D = inputX.getShape().dimension() == 2;
        NDArray x3d = is2D ? inputX.expandDims(0) : inputX;

        NDArray normX1 = norm1.forward(x3d);
        Shape shape = normX1.getShape();
        long batchSize = shape.get(0);
        long seqLen = shape.get(1);

        NDArray q = normX1.matMul(wQ).add(bQ);
        NDArray k = normX1.matMul(wK).add(bK);
        NDArray v = normX1.matMul(wV).add(bV);

        NDArray qHead = q.reshape(new Shape(batchSize, seqLen, numHeads, headDim)).transpose(0, 2, 1, 3);
        NDArray kHead = k.reshape(new Shape(batchSize, seqLen, numHeads, headDim)).transpose(0, 2, 1, 3);
        NDArray vHead = v.reshape(new Shape(batchSize, seqLen, numHeads, headDim)).transpose(0, 2, 1, 3);

        // Apply RoPE to Query and Key projections before cache storage
        if (cos != null && sin != null) {
            qHead = RopeEmbeddings.applyRope(qHead, cos, sin);
            kHead = RopeEmbeddings.applyRope(kHead, cos, sin);
        }

        // Update pre-allocated buffer in-place
        cacheBuffer.update(kHead, vHead);

        NDArray currentK = cacheBuffer.getActiveKeys();
        NDArray currentV = cacheBuffer.getActiveValues();

        NDArray scores = qHead.matMul(currentK.transpose(0, 1, 3, 2)).div(Math.sqrt(headDim));
        long totalSeqLen = currentK.getShape().get(2);

        CausalMaskGenerator.requireAttentionMask(causalMask, seqLen, totalSeqLen);
        if (causalMask != null) {
            scores = scores.add(causalMask);
        }

        NDArray attnWeights = scores.softmax(-1);
        NDArray context = attnWeights.matMul(currentV);

        NDArray concat = context.transpose(0, 2, 1, 3).reshape(new Shape(batchSize, seqLen, dModel));
        NDArray attnOut = concat.matMul(wO).add(bO);
        NDArray x1 = x3d.add(attnOut);

        NDArray normX2 = norm2.forward(x1);
        NDArray ffnOut = swiGLU(manager, normX2);
        NDArray out3d = x1.add(ffnOut);

        return is2D ? out3d.squeeze(0) : out3d;
    }

    private NDArray selfAttention(NDManager manager, NDArray x, NDArray causalMask, NDArray cos, NDArray sin) {
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

        if (cos != null && sin != null) {
            qHead = RopeEmbeddings.applyRope(qHead, cos, sin);
            kHead = RopeEmbeddings.applyRope(kHead, cos, sin);
        }

        NDArray scores = qHead.matMul(kHead.transpose(0, 1, 3, 2)).div(Math.sqrt(headDim));

        CausalMaskGenerator.requireAttentionMask(causalMask, seqLen, seqLen);
        if (causalMask != null) {
            scores = scores.add(causalMask);
        }

        NDArray attnWeights = scores.softmax(-1);
        NDArray context = attnWeights.matMul(vHead);

        NDArray concat = context.transpose(0, 2, 1, 3).reshape(is3D ? new Shape(batchSize, seqLen, dModel) : new Shape(seqLen, dModel));
        return concat.matMul(wO).add(bO);
    }

    private NDArray swiGLU(NDManager manager, NDArray x) {
        NDArray gate = x.matMul(wG).add(bG);
        NDArray up = x.matMul(wU);

        NDArray sigmoidGate = gate.neg().exp().add(1.0f).pow(-1.0f);
        NDArray swishGate = gate.mul(sigmoidGate);
        NDArray hidden = swishGate.mul(up);

        return hidden.matMul(wD).add(bD);
    }

    public List<NDArray> params() {
        List<NDArray> p = new ArrayList<>();
        // Attention parameters (8)
        p.add(wQ); p.add(bQ);
        p.add(wK); p.add(bK);
        p.add(wV); p.add(bV);
        p.add(wO); p.add(bO);
        
        // RMSNorm parameters (2)
        p.addAll(norm1.params());
        p.addAll(norm2.params());
        
        // SwiGLU parameters (5: wG, bG, wU, wD, bD)
        p.add(wG);
        if (bG != null) p.add(bG); // or add gate bias if instantiated
        p.add(wU);
        p.add(wD);
        if (bD != null) p.add(bD); // or add down bias if instantiated
        
        return p;
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

    /**
     * Core 7-argument implementation: handles NDManager, RoPE cos/sin, and KV-cache tensors.
     */
    public DecoderBlockResult forwardWithCache(NDManager manager, NDArray inputX, NDArray causalMask,
                                               NDArray cos, NDArray sin, NDArray pastK, NDArray pastV) {
        boolean is2D = inputX.getShape().dimension() == 2;
        NDArray x3d = is2D ? inputX.expandDims(0) : inputX;

        NDArray normX1 = norm1.forward(x3d);
        Shape shape = normX1.getShape();
        long batchSize = shape.get(0);
        long seqLen = shape.get(1);

        NDArray q = normX1.matMul(wQ).add(bQ);
        NDArray k = normX1.matMul(wK).add(bK);
        NDArray v = normX1.matMul(wV).add(bV);

        NDArray qHead = q.reshape(new Shape(batchSize, seqLen, numHeads, headDim)).transpose(0, 2, 1, 3);
        NDArray kHead = k.reshape(new Shape(batchSize, seqLen, numHeads, headDim)).transpose(0, 2, 1, 3);
        NDArray vHead = v.reshape(new Shape(batchSize, seqLen, numHeads, headDim)).transpose(0, 2, 1, 3);

        if (cos != null && sin != null) {
            qHead = RopeEmbeddings.applyRope(qHead, cos, sin);
            kHead = RopeEmbeddings.applyRope(kHead, cos, sin);
        }

        if ((pastK == null) != (pastV == null)) {
            throw new IllegalArgumentException("pastK and pastV must either both be null or both be provided");
        }
        NDArray currentK = appendToCache(pastK, kHead, "K", batchSize);
        NDArray currentV = appendToCache(pastV, vHead, "V", batchSize);

        NDArray scores = qHead.matMul(currentK.transpose(0, 1, 3, 2)).div(Math.sqrt(headDim));
        long totalSeqLen = currentK.getShape().get(2);

        CausalMaskGenerator.requireAttentionMask(causalMask, seqLen, totalSeqLen);
        if (causalMask != null) {
            scores = scores.add(causalMask);
        }

        NDArray attnWeights = scores.softmax(-1);
        NDArray context = attnWeights.matMul(currentV);

        NDArray concat = context.transpose(0, 2, 1, 3).reshape(new Shape(batchSize, seqLen, dModel));
        NDArray attnOut = concat.matMul(wO).add(bO);
        NDArray x1 = x3d.add(attnOut);

        NDArray normX2 = norm2.forward(x1);
        NDArray ffnOut = swiGLU(manager, normX2);
        NDArray out3d = x1.add(ffnOut);

        NDArray finalOut = is2D ? out3d.squeeze(0) : out3d;
        return new DecoderBlockResult(finalOut, currentK, currentV);
    }

    /**
     * 6-argument overload (without manager parameter).
     */
    public DecoderBlockResult forwardWithCache(NDArray inputX, NDArray causalMask, NDArray cos, NDArray sin,
                                               NDArray pastK, NDArray pastV) {
        return forwardWithCache(inputX.getManager(), inputX, causalMask, cos, sin, pastK, pastV);
    }

    /**
     * 5-argument overload with NDManager (without RoPE cos/sin).
     */
    public DecoderBlockResult forwardWithCache(NDManager manager, NDArray inputX, NDArray causalMask,
                                               NDArray pastK, NDArray pastV) {
        return forwardWithCache(manager, inputX, causalMask, null, null, pastK, pastV);
    }

    /**
     * 4-argument overload (without manager and without RoPE cos/sin).
     */
    public DecoderBlockResult forwardWithCache(NDArray inputX, NDArray causalMask, NDArray pastK, NDArray pastV) {
        return forwardWithCache(inputX.getManager(), inputX, causalMask, null, null, pastK, pastV);
    }

    // 1. Core 5-argument forward pass (with manager and RoPE cos/sin)
    public NDArray forward(NDManager manager, NDArray inputX, NDArray causalMask, NDArray cos, NDArray sin) {
        NDArray normX1 = norm1.forward(inputX);
        NDArray attnOut = selfAttention(manager, normX1, causalMask, cos, sin);
        NDArray x1 = inputX.add(attnOut);

        NDArray normX2 = norm2.forward(x1);
        NDArray ffnOut = swiGLU(manager, normX2);
        return x1.add(ffnOut);
    }

    // 2. 4-argument forward pass (uses inputX manager with RoPE cos/sin)
    public NDArray forward(NDArray inputX, NDArray causalMask, NDArray cos, NDArray sin) {
        return forward(inputX.getManager(), inputX, causalMask, cos, sin);
    }

    // 3. 3-argument forward pass with manager (no RoPE cos/sin)
    public NDArray forward(NDManager manager, NDArray inputX, NDArray causalMask) {
        return forward(manager, inputX, causalMask, null, null);
    }

    // 4. 2-argument forward pass (no manager, no RoPE cos/sin)
    public NDArray forward(NDArray inputX, NDArray causalMask) {
        return forward(inputX.getManager(), inputX, causalMask, null, null);
    }
}