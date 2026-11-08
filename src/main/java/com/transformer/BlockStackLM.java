package com.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.decoder.DecoderBlock;

import java.util.ArrayList;
import java.util.List;

public class BlockStackLM {

    private final int numLayers;
    private final int dModel;
    private final int numHeads;
    private final int headDim;
    private final int vocabSize;
    private final List<DecoderBlock> layers;
    private final RMSNorm finalNorm;
    private final NDArray lmHead; // Linear projection: [dModel, vocabSize]

    public BlockStackLM(NDManager manager, int numLayers, int dModel, int numHeads, int dFF, int vocabSize, float eps) {
        this.numLayers = numLayers;
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.headDim = dModel / numHeads;
        this.vocabSize = vocabSize;

        this.layers = new ArrayList<>(numLayers);
        for (int i = 0; i < numLayers; i++) {
            layers.add(new DecoderBlock(manager, dModel, numHeads, dFF, eps));
        }

        this.finalNorm = new RMSNorm(manager, dModel, eps);
        
        float scale = (float) Math.sqrt(2.0 / dModel);
        this.lmHead = manager.randomUniform(-scale, scale, new Shape(dModel, vocabSize), DataType.FLOAT32);
        this.lmHead.setRequiresGradient(true);
    }

    /**
     * Initializes pre-allocated KV Cache buffers for all transformer layers.
     */
    public List<KVCacheBuffer> createLayerBuffers(NDManager manager, int batchSize, int maxSeqLen) {
        List<KVCacheBuffer> buffers = new ArrayList<>(numLayers);
        for (int i = 0; i < numLayers; i++) {
            buffers.add(new KVCacheBuffer(manager, batchSize, numHeads, maxSeqLen, headDim));
        }
        return buffers;
    }

    /**
     * Full sequence forward pass (non-cached, standard O(N^2) evaluation).
     */
    public NDArray forward(NDManager manager, NDArray inputX, NDArray causalMask, NDArray cos, NDArray sin) {
        NDArray hidden = inputX;
        for (DecoderBlock layer : layers) {
            hidden = layer.forward(manager, hidden, causalMask, cos, sin);
        }
        NDArray normed = finalNorm.forward(hidden);
        return normed.matMul(lmHead);
    }

    public NDArray forward(NDManager manager, NDArray inputX, NDArray causalMask) {
        return forward(manager, inputX, causalMask, null, null);
    }

    public NDArray forward(NDArray inputX, NDArray causalMask, NDArray cos, NDArray sin) {
        return forward(inputX.getManager(), inputX, causalMask, cos, sin);
    }

    public NDArray forward(NDArray inputX, NDArray causalMask) {
        return forward(inputX.getManager(), inputX, causalMask, null, null);
    }

    /**
     * Step-by-step or prompt prefill forward pass using static layer-wise KVCacheBuffers.
     *
     * @param inputX Hidden states tensor [batchSize, stepSeqLen, dModel] or [stepSeqLen, dModel]
     * @param causalMask Attention mask tensor
     * @param cos Positional cosine embeddings
     * @param sin Positional sine embeddings
     * @param layerBuffers Pre-allocated KV buffers for each layer
     * @return Logits tensor over vocabulary [batchSize, stepSeqLen, vocabSize] or [stepSeqLen, vocabSize]
     */
    public NDArray forwardWithBuffer(NDManager manager, NDArray inputX, NDArray causalMask,
                                    NDArray cos, NDArray sin, List<KVCacheBuffer> layerBuffers) {
        if (layerBuffers.size() != numLayers) {
            throw new IllegalArgumentException(
                String.format("Expected %d layer buffers but got %d", numLayers, layerBuffers.size())
            );
        }

        NDArray hidden = inputX;
        for (int i = 0; i < numLayers; i++) {
            DecoderBlock layer = layers.get(i);
            KVCacheBuffer buffer = layerBuffers.get(i);
            hidden = layer.forwardWithBuffer(manager, hidden, causalMask, cos, sin, buffer);
        }

        NDArray normed = finalNorm.forward(hidden);
        return normed.matMul(lmHead);
    }

    public List<NDArray> params() {
        List<NDArray> params = new ArrayList<>();
        for (DecoderBlock layer : layers) {
            params.addAll(layer.params());
        }
        params.addAll(finalNorm.params());
        params.add(lmHead);
        return params;
    }

    public int getNumLayers() {
        return numLayers;
    }

    public int getDModel() {
        return dModel;
    }

    public int getVocabSize() {
        return vocabSize;
    }
}