package com.decoder;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
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

    public DecoderBlockResult forwardWithCache(NDArray inputX, NDArray causalMask, NDArray pastK, NDArray pastV, NDArray arg5, NDArray arg6) {
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

    // 2. Safe Past Key Cache Formatting and Concatenation
    NDArray currentK;
    if (pastK != null) {
        NDArray formattedPastK = pastK;
        if (pastK.getShape().dimension() != 4) {
            long totalElements = pastK.getShape().size();
            long fullHeadSize = batchSize * numHeads * headDim;

            if (totalElements >= fullHeadSize) {
                long pastSeq = totalElements / fullHeadSize;
                formattedPastK = pastK.reshape(new Shape(batchSize, pastSeq, numHeads, headDim))
                                      .transpose(0, 2, 1, 3);
            } else {
                // Handle flat/single-head past cache seed [batchSize, pastSeq, headDim]
                long pastSeq = Math.max(1, totalElements / (batchSize * headDim));
                formattedPastK = pastK.reshape(new Shape(batchSize, pastSeq, 1, headDim))
                                      .transpose(0, 2, 1, 3)
                                      .repeat(1, numHeads);
            }
        }
        currentK = formattedPastK.concat(kHead, 2);
    } else {
        currentK = kHead;
    }

    // 3. Safe Past Value Cache Formatting and Concatenation
    NDArray currentV;
    if (pastV != null) {
        NDArray formattedPastV = pastV;
        if (pastV.getShape().dimension() != 4) {
            long totalElements = pastV.getShape().size();
            long fullHeadSize = batchSize * numHeads * headDim;

            if (totalElements >= fullHeadSize) {
                long pastSeq = totalElements / fullHeadSize;
                formattedPastV = pastV.reshape(new Shape(batchSize, pastSeq, numHeads, headDim))
                                      .transpose(0, 2, 1, 3);
            } else {
                long pastSeq = Math.max(1, totalElements / (batchSize * headDim));
                formattedPastV = pastV.reshape(new Shape(batchSize, pastSeq, 1, headDim))
                                      .transpose(0, 2, 1, 3)
                                      .repeat(1, numHeads);
            }
        }
        currentV = formattedPastV.concat(vHead, 2);
    } else {
        currentV = vHead;
    }

    // 4. Scaled Dot-Product Attention over dynamic sequence length
    NDArray scores = qHead.matMul(currentK.transpose(0, 1, 3, 2)).div(Math.sqrt(headDim));
    long totalSeqLen = currentK.getShape().get(2);

    // 5. Causal Mask Alignment for Incremental Step Decoding
    if (causalMask != null) {
        Shape maskShape = causalMask.getShape();
        NDArray activeMask = null;

        if (maskShape.dimension() == 4) {
            long maskQ = maskShape.get(2);
            long maskK = maskShape.get(3);
            long qStart = Math.max(0, maskQ - seqLen);
            long kEnd = Math.min(maskK, totalSeqLen);

            if (qStart < maskQ && kEnd > 0) {
                String pattern = String.format(":, :, %d:%d, 0:%d", qStart, maskQ, kEnd);
                activeMask = causalMask.get(new ai.djl.ndarray.index.NDIndex(pattern));
            }
        } else if (maskShape.dimension() == 2) {
            long maskQ = maskShape.get(0);
            long maskK = maskShape.get(1);
            long qStart = Math.max(0, maskQ - seqLen);
            long kEnd = Math.min(maskK, totalSeqLen);

            if (qStart < maskQ && kEnd > 0) {
                String pattern = String.format("%d:%d, 0:%d", qStart, maskQ, kEnd);
                activeMask = causalMask.get(new ai.djl.ndarray.index.NDIndex(pattern));
            }
        }

        if (activeMask != null) {
            Shape aShape = activeMask.getShape();
            long aQ = aShape.get(aShape.dimension() - 2);
            long aK = aShape.get(aShape.dimension() - 1);

            if (aQ == scores.getShape().get(2) && aK == scores.getShape().get(3)) {
                scores = scores.add(activeMask);
            }
        }
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