package com.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.decoder.DecoderBlock;

import java.util.ArrayList;
import java.util.List;

public class BlockStackLM {

    private final int dModel;
    private final int numHeads;
    private final int numLayers;
    private final int dFF;
    private final int vocabSize;
    private final float eps;

    private final List<DecoderBlock> layers;
    private final RMSNorm finalNorm;
    private final NDArray lmHead; // Output vocabulary projection matrix [dModel, vocabSize]

    public BlockStackLM(NDManager manager, int dModel, int numHeads, int numLayers, int dFF, int vocabSize, float eps) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.numLayers = numLayers;
        this.dFF = dFF;
        this.vocabSize = vocabSize;
        this.eps = eps;

        this.layers = new ArrayList<>();
        for (int i = 0; i < numLayers; i++) {
            layers.add(new DecoderBlock(manager, dModel, numHeads, dFF, eps));
        }

        this.finalNorm = new RMSNorm(manager, dModel, eps);

        float scale = (float) Math.sqrt(2.0 / dModel);
        this.lmHead = manager.randomUniform(-scale, scale, new Shape(dModel, vocabSize), DataType.FLOAT32);

        // Ensure all instantiated model parameters enable autograd tracking by default
        for (NDArray param : params()) {
            param.setRequiresGradient(true);
        }
    }

    /**
     * Standard 3-argument forward pass: accepts inputs and passes through all decoder blocks, 
     * final RMSNorm, and language modeling head projection.
     *
     * @param manager    NDManager context for intermediate tensors
     * @param inputX     Embedded input tensor [batchSize, seqLen, dModel] or [seqLen, dModel]
     * @param causalMask Causal attention mask tensor
     * @return Logits projected to vocabulary dimension [batchSize, seqLen, vocabSize]
     */
    public NDArray forward(NDManager manager, NDArray inputX, NDArray causalMask) {
        NDArray x = inputX;

        // Pass through stacked decoder blocks
        for (DecoderBlock block : layers) {
            x = block.forward(manager, x, causalMask);
        }

        // Apply final Pre-LM RMSNorm layer
        NDArray normX = finalNorm.forward(x);

        // Project hidden state to vocabulary dimension logits
        return normX.matMul(lmHead);
    }

    /**
     * Overloaded 4-argument forward pass for API compatibility across test runners and pipelines.
     */
    public NDArray forward(NDManager manager, NDArray inputX, NDArray causalMask, NDArray extraArg) {
        return forward(manager, inputX, causalMask);
    }

    /**
     * Collects and exposes all trainable parameters across the stack, final normalization layer, 
     * and output language modeling head for the optimizer and gradient collector.
     *
     * @return List of all trainable NDArray parameters
     */
    public List<NDArray> params() {
        List<NDArray> allParams = new ArrayList<>();

        // 1. Decoder Stack Parameters (14 parameters per layer: 7 Attention + 4 RMSNorm + 3 SwiGLU)
        for (DecoderBlock block : layers) {
            allParams.addAll(block.params());
        }

        // 2. Final RMSNorm Layer Parameters (gamma & beta)
        allParams.addAll(finalNorm.params());

        // 3. Language Model Output Head Projection Weight
        allParams.add(lmHead);

        return allParams;
    }

    public int getDModel() {
        return dModel;
    }

    public int getNumHeads() {
        return numHeads;
    }

    public int getNumLayers() {
        return numLayers;
    }

    public int getVocabSize() {
        return vocabSize;
    }
}