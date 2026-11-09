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

/**
 * One Pre-RMSNorm decoder layer: causal multi-head attention with RoPE, then a SwiGLU feed-forward block.
 * <p>
 * Every public entry point ends in the same {@link #run} method, so training, full scoring and cached decoding
 * cannot drift apart. Rules that used to be easy to break silently are now enforced:
 * <ul>
 *   <li>RoPE is always applied. The RoPE tables must end with [tokensProcessed, headDim]; null or wrongly
 *       sized tables throw (there is no "no RoPE" mode).</li>
 *   <li>The attention mask must be exactly [queries, keys] (see {@link CausalMaskGenerator#requireAttentionMask}).</li>
 *   <li>Cached decoding uses a {@link KVCacheBuffer}: the new keys (already rotated) are written in place and
 *       attention reads the active slice.</li>
 * </ul>
 */
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

    // ------------------------------------------------------------------ full passes (training / scoring)

    /**
     * Full pass over a whole sequence. cos/sin must end with [seqLen, headDim] (rotations for positions
     * 0..seqLen-1, e.g. from {@link RopeEmbeddings#computeCosSin} sliced to seqLen rows).
     */
    public NDArray forward(NDArray inputX, NDArray causalMask, NDArray cos, NDArray sin) {
        return run(inputX, causalMask, cos, sin, null);
    }

    /** Same as {@link #forward(NDArray, NDArray, NDArray, NDArray)}; the manager is not needed any more. */
    public NDArray forward(NDManager manager, NDArray inputX, NDArray causalMask, NDArray cos, NDArray sin) {
        return run(inputX, causalMask, cos, sin, null);
    }

    /**
     * Full pass that builds its own RoPE tables (positions 0..seqLen-1) on {@code manager}. In a layer loop build
     * the tables once and use the explicit form instead.
     */
    public NDArray forward(NDManager manager, NDArray inputX, NDArray causalMask) {
        NDArray[] rope = RopeEmbeddings.computeCosSin(manager, (int) sequenceLength(inputX), headDim);
        return run(inputX, causalMask, rope[0], rope[1], null);
    }

    // ------------------------------------------------------------------ cached decoding

    /**
     * Cached step or prefill. cos/sin must be the rotations for the NEW tokens only (rows
     * [cache.getCurrentLength(), cache.getCurrentLength() + newTokens) of the full tables, see
     * {@link RopeEmbeddings#slice}). The mask must be [newTokens, cached + newTokens]
     * ({@link CausalMaskGenerator#createCachedMask}); null is allowed for a single new token.
     */
    public NDArray forwardWithBuffer(NDManager manager, NDArray inputX, NDArray causalMask,
                                     NDArray cos, NDArray sin, KVCacheBuffer cacheBuffer) {
        if (cacheBuffer == null) {
            throw new IllegalArgumentException("A KVCacheBuffer is required");
        }
        return run(inputX, causalMask, cos, sin, cacheBuffer);
    }

    /** Cached step or prefill that works out the RoPE rows from the number of tokens already in the cache. */
    public NDArray forwardWithBuffer(NDManager manager, NDArray inputX, NDArray causalMask, KVCacheBuffer cacheBuffer) {
        int newTokens = (int) sequenceLength(inputX);
        int start = cacheBuffer.getCurrentLength();
        NDArray[] all = RopeEmbeddings.computeCosSin(manager, start + newTokens, headDim);
        NDArray[] rows = RopeEmbeddings.slice(all[0], all[1], start, newTokens);
        return run(inputX, causalMask, rows[0], rows[1], cacheBuffer);
    }

    // ------------------------------------------------------------------ the single implementation

    private NDArray run(NDArray inputX, NDArray causalMask, NDArray cos, NDArray sin, KVCacheBuffer cache) {
        boolean is2D = inputX.getShape().dimension() == 2;
        NDArray x = is2D ? inputX.expandDims(0) : inputX;

        NDArray afterAttention = x.add(attention(norm1.forward(x), causalMask, cos, sin, cache));
        NDArray out = afterAttention.add(swiGLU(norm2.forward(afterAttention)));

        return is2D ? out.squeeze(0) : out;
    }

    /** normX: [batch, seqLen, dModel] -> attention output [batch, seqLen, dModel] (before the residual add). */
    private NDArray attention(NDArray normX, NDArray causalMask, NDArray cos, NDArray sin, KVCacheBuffer cache) {
        Shape shape = normX.getShape();
        long batchSize = shape.get(0);
        long seqLen = shape.get(1);

        NDArray qHead = splitHeads(normX.matMul(wQ).add(bQ), batchSize, seqLen);
        NDArray kHead = splitHeads(normX.matMul(wK).add(bK), batchSize, seqLen);
        NDArray vHead = splitHeads(normX.matMul(wV).add(bV), batchSize, seqLen);

        // Validate everything BEFORE the cache is touched: a rejected call must leave the cache exactly as it was.
        long totalKeys = (cache == null ? 0 : cache.getCurrentLength()) + seqLen;
        CausalMaskGenerator.requireAttentionMask(causalMask, seqLen, totalKeys);
        requireRopeTable(cos, "cos", seqLen);
        requireRopeTable(sin, "sin", seqLen);

        // RoPE on the NEW queries/keys only. Keys already in the cache were rotated when they were stored.
        qHead = RopeEmbeddings.applyRope(qHead, cos, sin);
        kHead = RopeEmbeddings.applyRope(kHead, cos, sin);

        NDArray keys = kHead;
        NDArray values = vHead;
        if (cache != null) {
            cache.update(kHead, vHead);
            keys = cache.getActiveKeys();
            values = cache.getActiveValues();
        }

        NDArray scores = qHead.matMul(keys.transpose(0, 1, 3, 2)).div(Math.sqrt(headDim));
        if (causalMask != null) {
            scores = scores.add(causalMask);
        }

        NDArray context = scores.softmax(-1).matMul(values);
        return context.transpose(0, 2, 1, 3).reshape(new Shape(batchSize, seqLen, dModel)).matMul(wO).add(bO);
    }

    private NDArray splitHeads(NDArray projected, long batchSize, long seqLen) {
        return projected.reshape(new Shape(batchSize, seqLen, numHeads, headDim)).transpose(0, 2, 1, 3);
    }

    private NDArray swiGLU(NDArray x) {
        NDArray gate = x.matMul(wG).add(bG);
        NDArray up = x.matMul(wU);

        NDArray sigmoidGate = gate.neg().exp().add(1.0f).pow(-1.0f);
        NDArray hidden = gate.mul(sigmoidGate).mul(up);

        return hidden.matMul(wD).add(bD);
    }

    private void requireRopeTable(NDArray table, String name, long seqLen) {
        if (table == null) {
            throw new IllegalArgumentException("RoPE table '" + name + "' is required: this block always applies RoPE");
        }
        Shape s = table.getShape();
        int r = s.dimension();
        if (r < 2 || s.get(r - 2) != seqLen || s.get(r - 1) != headDim) {
            throw new IllegalArgumentException("RoPE table '" + name + "' must end with [" + seqLen + ", " + headDim
                    + "] (the rotations for exactly the tokens being processed) but was " + s
                    + ". For cached decoding take the rows of the new positions with RopeEmbeddings.slice(...).");
        }
    }

    private static long sequenceLength(NDArray x) {
        Shape s = x.getShape();
        return s.get(s.dimension() - 2);
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
        p.add(bG);
        p.add(wU);
        p.add(wD);
        p.add(bD);

        return p;
    }

    public int getHeadDim() {
        return headDim;
    }

    public int getNumHeads() {
        return numHeads;
    }
}
