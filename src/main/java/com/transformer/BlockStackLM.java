package com.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.decoder.DecoderBlock;

import java.util.ArrayList;
import java.util.List;

/**
 * Decoder-only language model: token embeddings -> N x {@link DecoderBlock} -> final RMSNorm -> output head.
 * <p>
 * The embedding table lives INSIDE the model and is tied with the output head (logits = hidden x embedding^T),
 * so a checkpoint of {@link #params()} is the whole model and every parameter is trained. Look-ups are done as
 * one-hot x table, which keeps the gradient path to the table intact (a copy through a Java array does not).
 * <p>
 * Pass a per-step sub-manager as {@code manager}: every intermediate tensor and the transposed head are attached
 * to it, so closing it frees them.
 */
public class BlockStackLM {

    private final LMConfig config;
    private final NDArray embedding; // [vocabSize, dModel], also the output projection
    private final List<DecoderBlock> layers;
    private final RMSNorm finalNorm;

    /** Legacy initialisation of the blocks (Xavier-style uniform); see the other constructor. */
    public BlockStackLM(NDManager manager, LMConfig config) {
        this(manager, config, 0f);
    }

    /**
     * @param weightStd 0 keeps the legacy initialisation. &gt; 0 selects GPT-style initialisation: normal(0, weightStd) for
     *                  all weights, and weightStd / sqrt(2 * numLayers) for the matrices that write into the residual
     *                  stream, so activations stay in range however deep the stack is. 0.02 is the usual value.
     */
    public BlockStackLM(NDManager manager, LMConfig config, float weightStd) {
        this.config = config;

        this.embedding = manager.randomNormal(0.0f, 0.02f, new Shape(config.vocabSize(), config.dModel()), DataType.FLOAT32);
        this.embedding.setRequiresGradient(true);

        this.layers = new ArrayList<>(config.numLayers());
        for (int i = 0; i < config.numLayers(); i++) {
            float outputStd = weightStd > 0f ? weightStd / (float) Math.sqrt(2.0 * config.numLayers()) : 0f;
            layers.add(new DecoderBlock(manager, config.dModel(), config.numHeads(), config.dFF(), config.eps(),
                    weightStd, outputStd));
        }
        this.finalNorm = new RMSNorm(manager, config.dModel(), config.eps());
    }

    // ------------------------------------------------------------------ embeddings

    /**
     * Looks up token embeddings. ids: integer ids (any numeric dtype) of shape [seqLen] or [batch, seqLen]
     * -> [seqLen, dModel] or [batch, seqLen, dModel]. Differentiable w.r.t. the embedding table.
     */
    public NDArray embed(NDArray tokenIds) {
        // One-hot on a flat [N] vector (the path every model in this repo uses), then restore the original shape.
        long[] idDims = tokenIds.getShape().getShape();
        NDArray flat = tokenIds.reshape(tokenIds.size()).toType(DataType.INT64, false);
        NDArray vectors = flat.oneHot(config.vocabSize()).matMul(embedding); // [N, dModel]

        long[] outDims = java.util.Arrays.copyOf(idDims, idDims.length + 1);
        outDims[idDims.length] = config.dModel();
        return vectors.reshape(outDims);
    }

    // ------------------------------------------------------------------ full passes

    /** Hidden states [.., seqLen, dModel] -> logits. cos/sin must end with [seqLen, headDim]. */
    public NDArray forward(NDManager manager, NDArray inputX, NDArray causalMask, NDArray cos, NDArray sin) {
        NDArray hidden = inputX;
        for (DecoderBlock layer : layers) {
            hidden = layer.forward(hidden, causalMask, cos, sin);
        }
        return project(manager, hidden);
    }

    /** Hidden states -> logits, with RoPE tables for positions 0..seqLen-1 built once and shared by all layers. */
    public NDArray forward(NDManager manager, NDArray inputX, NDArray causalMask) {
        Shape s = inputX.getShape();
        NDArray[] rope = RopeEmbeddings.computeCosSin(manager, (int) s.get(s.dimension() - 2), config.headDim());
        return forward(manager, inputX, causalMask, rope[0], rope[1]);
    }

    /** Token ids -> logits (embedding look-up included). The usual entry point for training. */
    public NDArray forwardIds(NDManager manager, NDArray tokenIds, NDArray causalMask) {
        return forward(manager, embed(tokenIds), causalMask);
    }

    // ------------------------------------------------------------------ cached decoding

    /** One pre-allocated KV buffer per layer. */
    public List<KVCacheBuffer> createLayerBuffers(NDManager manager, int batchSize, int maxSeqLen) {
        List<KVCacheBuffer> buffers = new ArrayList<>(config.numLayers());
        for (int i = 0; i < config.numLayers(); i++) {
            buffers.add(new KVCacheBuffer(manager, batchSize, config.numHeads(), maxSeqLen, config.headDim()));
        }
        return buffers;
    }

    /**
     * Prefill or single-step decoding with explicit RoPE rows for the new tokens (see
     * {@link RopeEmbeddings#slice}) and the matching cached mask ({@link CausalMaskGenerator#createCachedMask}).
     */
    public NDArray forwardWithBuffer(NDManager manager, NDArray inputX, NDArray causalMask,
                                     NDArray cos, NDArray sin, List<KVCacheBuffer> layerBuffers) {
        if (layerBuffers.size() != config.numLayers()) {
            throw new IllegalArgumentException(
                    String.format("Expected %d layer buffers but got %d", config.numLayers(), layerBuffers.size()));
        }
        NDArray hidden = inputX;
        for (int i = 0; i < config.numLayers(); i++) {
            hidden = layers.get(i).forwardWithBuffer(manager, hidden, causalMask, cos, sin, layerBuffers.get(i));
        }
        return project(manager, hidden);
    }

    /** Prefill or single-step decoding; the RoPE rows are taken from the number of tokens already cached. */
    public NDArray forwardWithBuffer(NDManager manager, NDArray inputX, NDArray causalMask,
                                     List<KVCacheBuffer> layerBuffers) {
        int start = layerBuffers.get(0).getCurrentLength();
        for (KVCacheBuffer buffer : layerBuffers) {
            if (buffer.getCurrentLength() != start) {
                throw new IllegalStateException("Layer caches are out of sync: " + start + " vs " + buffer.getCurrentLength());
            }
        }
        Shape s = inputX.getShape();
        int newTokens = (int) s.get(s.dimension() - 2);
        NDArray[] all = RopeEmbeddings.computeCosSin(manager, start + newTokens, config.headDim());
        NDArray[] rows = RopeEmbeddings.slice(all[0], all[1], start, newTokens);
        return forwardWithBuffer(manager, inputX, causalMask, rows[0], rows[1], layerBuffers);
    }

    // ------------------------------------------------------------------ plumbing

    private NDArray project(NDManager scope, NDArray hidden) {
        NDArray embeddingT = embedding.transpose();
        embeddingT.attach(scope); // otherwise one transposed copy per call stays alive on the base manager
        return finalNorm.forward(hidden).matMul(embeddingT);
    }

    /** All trainable tensors, embedding table last. This is exactly what a checkpoint has to contain. */
    /** One name per entry of {@link #params()}, in the same order. */
    public List<String> paramNames() {
        List<String> names = new ArrayList<>();
        for (int i = 0; i < layers.size(); i++) {
            for (String name : layers.get(i).paramNames()) {
                names.add("layer" + i + "." + name);
            }
        }
        names.add("finalNorm.gamma");
        names.add("embedding");
        return names;
    }

    public List<NDArray> params() {
        List<NDArray> params = new ArrayList<>();
        for (DecoderBlock layer : layers) {
            params.addAll(layer.params());
        }
        params.addAll(finalNorm.params());
        params.add(embedding);
        return params;
    }

    public LMConfig getConfig() {
        return config;
    }

    public NDArray getEmbedding() {
        return embedding;
    }

    public int getNumLayers() {
        return config.numLayers();
    }

    public int getDModel() {
        return config.dModel();
    }

    public int getHeadDim() {
        return config.headDim();
    }

    public int getVocabSize() {
        return config.vocabSize();
    }
}
