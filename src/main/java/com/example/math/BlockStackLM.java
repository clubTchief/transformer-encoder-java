package com.example.math;

import com.transformer.LMHead;
import com.decoder.DecoderBlock;
import com.encoder.ModernizedHierarchicalEncoderPipeline;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.List;

public class BlockStackLM {

    private final List<DecoderBlock> blocks;
    private final LMHead lmHead;
    private final int dModel;
    private final int numHeads;
    private final int numLayers;
    private final int vocabSize;

    public BlockStackLM(NDManager manager, int dModel, int numHeads, int numLayers, int dFF, int vocabSize, float eps) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.numLayers = numLayers;
        this.vocabSize = vocabSize;

        this.blocks = new ArrayList<>();
        for (int i = 0; i < numLayers; i++) {
            this.blocks.add(new DecoderBlock(manager, dModel, numHeads, dFF, eps));
        }

        this.lmHead = new LMHead(manager, dModel, vocabSize);
    }

    /**
     * Executes forward pass across all DecoderBlocks and projects hidden states to logits.
     * Input X shape: [seqLen, dModel]
     * Output shape: [seqLen, vocabSize]
     */
    public NDArray forward(NDManager manager, NDArray X, NDArray causalMask) {
        long seqLen = X.getShape().get(0);
        int dK = dModel / numHeads;

        // Pre-compute RoPE frequencies once per pass
        NDArray[] ropeFreqs = ModernizedHierarchicalEncoderPipeline.precomputeRoPEFrequencies(manager, seqLen, dK);
        NDArray cos = ropeFreqs[0];
        NDArray sin = ropeFreqs[1];

        NDArray currentX = X;
        for (DecoderBlock block : blocks) {
            currentX = block.forward(currentX, causalMask, cos, sin);
        }

        return lmHead.forward(currentX);
    }

    /**
     * Numerically stable Cross-Entropy Loss computation for a sequence of target token IDs.
     * Prevents NaN values through max-logit shift and log-softmax epsilon bounds.
     */
    public static float computeLoss(NDArray logits, int[] targetTokenIds) {
        Shape shape = logits.getShape();
        long seqLen = shape.get(0);
        long vocabSize = shape.get(1);

        float[] flatLogits = logits.toFloatArray();
        float totalLoss = 0.0f;

        for (int t = 0; t < seqLen; t++) {
            int targetId = targetTokenIds[t];
            int offset = (int) (t * vocabSize);

            // 1. Numerically stable max logit shift
            float maxLogit = Float.NEGATIVE_INFINITY;
            for (int v = 0; v < vocabSize; v++) {
                float val = flatLogits[offset + v];
                if (val > maxLogit) maxLogit = val;
            }

            // 2. Compute denominator (sum of exps)
            float sumExp = 0.0f;
            for (int v = 0; v < vocabSize; v++) {
                sumExp += (float) Math.exp(flatLogits[offset + v] - maxLogit);
            }

            // 3. Compute target probability with safety bounds
            float targetLogit = flatLogits[offset + targetId];
            float prob = (float) Math.exp(targetLogit - maxLogit) / sumExp;
            
            // Clamp prob to prevent log(0.0) -> NaN/Infinity
            prob = Math.max(prob, 1e-7f);
            totalLoss += -(float) Math.log(prob);
        }

        return totalLoss / seqLen;
    }

    /**
     * Returns all registered trainable parameters across all blocks and the LM Head.
     */
    public List<NDArray> params() {
        List<NDArray> allParams = new ArrayList<>();
        for (DecoderBlock block : blocks) {
            allParams.addAll(block.params());
        }
        allParams.addAll(lmHead.params());
        return allParams;
    }

    public List<DecoderBlock> getBlocks() {
        return blocks;
    }

    public LMHead getLmHead() {
        return lmHead;
    }
}