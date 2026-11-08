package com.example.story;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.transformer.CausalMaskGenerator;
import com.decoder.DecoderBlock;
import com.transformer.LayerNorm;
import com.encoder.ModernizedHierarchicalEncoderPipeline;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * A language model built from the repo's own {@link DecoderBlock} stack:
 * learned token embeddings -> N x DecoderBlock -> final LayerNorm -> output head tied to the embeddings.
 *
 * One sequence at a time: ids INT32 [T] in, logits FLOAT32 [T, V] out. (TinyGpt is the batched sibling.)
 * Pass a per-step sub-manager as {@code scope}; every intermediate is owned by it, so closing it frees them.
 */
public final class BlockStackLM {

    private final GptConfig cfg;
    private final NDArray tokEmb; // [V, d], also the output head (transposed)
    private final List<DecoderBlock> blocks = new ArrayList<>();
    private final LayerNorm finalNorm;

    public BlockStackLM(NDManager manager, GptConfig cfg, long seed) {
        this.cfg = cfg;
        Engine.getInstance().setRandomSeed((int) seed);
        tokEmb = manager.randomNormal(0.0f, 0.02f, new Shape(cfg.vocabSize(), cfg.dModel()), DataType.FLOAT32);
        tokEmb.setRequiresGradient(true);
        for (int l = 0; l < cfg.numLayers(); l++) {
            blocks.add(new DecoderBlock(manager, cfg.dModel(), cfg.numHeads(), cfg.dFF(), cfg.eps()));
        }
        finalNorm = new LayerNorm(manager, cfg.dModel(), cfg.eps());
    }

    public GptConfig config() {
        return cfg;
    }

    /** All trainable tensors in a fixed order (optimizer + checkpoints). */
    public List<NDArray> params() {
        List<NDArray> p = new ArrayList<>();
        p.add(tokEmb);
        for (DecoderBlock b : blocks) p.addAll(b.params());
        p.add(finalNorm.getGamma());
        p.add(finalNorm.getBeta());
        return p;
    }

    /** ids: INT32 [T]  ->  logits: FLOAT32 [T, V]. */
    public NDArray forward(NDManager scope, NDArray ids) {
        long T = ids.getShape().get(0);
        if (T > cfg.contextLength()) {
            throw new IllegalArgumentException("T=" + T + " exceeds context length " + cfg.contextLength());
        }
        NDArray mask = CausalMaskGenerator.createCausalMask(scope, T).squeeze(0).squeeze(0);
        NDArray[] rope = ModernizedHierarchicalEncoderPipeline.precomputeRoPEFrequencies(scope, T, cfg.headDim());
        NDArray embT = tokEmb.transpose();
        embT.attach(scope);

        // Embedding lookup as one-hot @ table: differentiable w.r.t. the table.
        NDArray x = ids.toType(DataType.INT64, false).oneHot(cfg.vocabSize()).matMul(tokEmb); // [T, d]
        for (DecoderBlock block : blocks) {
            x = block.forward(x, mask, rope[0], rope[1]);
        }
        return finalNorm.forward(x).matMul(embT); // weight tying
    }

    /** Mean next-token cross-entropy. logits [T, V], targets INT32 [T]. */
    public NDArray loss(NDArray logits, NDArray targets) {
        long n = logits.getShape().get(0);
        NDArray logProbs = logits.logSoftmax(-1);
        NDArray oneHot = targets.toType(DataType.INT64, false).oneHot(cfg.vocabSize());
        return logProbs.mul(oneHot).sum().neg().div((float) n);
    }

    /** Sampling front-end for StoryGenerator. */
    public StoryGenerator generator(NDManager manager, CharTokenizer tok) {
        return new StoryGenerator(manager, cfg.contextLength(), (scope, window) -> {
            NDArray out = forward(scope, scope.create(window)); // [T, V]
            return out.get(new NDIndex((window.length - 1) + ", :")).toFloatArray();
        }, tok);
    }

    // ------------------------------------------------------------------ checkpoints

    public void save(Path dir) throws IOException {
        Files.createDirectories(dir);
        Files.write(dir.resolve("stack.params"), new NDList(params()).encode());
        Properties p = new Properties();
        p.setProperty("vocabSize", String.valueOf(cfg.vocabSize()));
        p.setProperty("dModel", String.valueOf(cfg.dModel()));
        p.setProperty("numHeads", String.valueOf(cfg.numHeads()));
        p.setProperty("numLayers", String.valueOf(cfg.numLayers()));
        p.setProperty("dFF", String.valueOf(cfg.dFF()));
        p.setProperty("contextLength", String.valueOf(cfg.contextLength()));
        p.setProperty("eps", String.valueOf(cfg.eps()));
        try (Writer w = Files.newBufferedWriter(dir.resolve("stack.cfg"))) {
            p.store(w, "BlockStackLM config");
        }
    }

    public static BlockStackLM load(NDManager manager, Path dir) throws IOException {
        Properties p = new Properties();
        try (Reader r = Files.newBufferedReader(dir.resolve("stack.cfg"))) {
            p.load(r);
        }
        GptConfig cfg = new GptConfig(
                Integer.parseInt(p.getProperty("vocabSize")), Integer.parseInt(p.getProperty("dModel")),
                Integer.parseInt(p.getProperty("numHeads")), Integer.parseInt(p.getProperty("numLayers")),
                Integer.parseInt(p.getProperty("dFF")), Integer.parseInt(p.getProperty("contextLength")),
                Float.parseFloat(p.getProperty("eps")));
        BlockStackLM model = new BlockStackLM(manager, cfg, 0);
        try (NDList saved = NDList.decode(manager, Files.readAllBytes(dir.resolve("stack.params")))) {
            List<NDArray> dst = model.params();
            if (saved.size() != dst.size()) {
                throw new IOException("Checkpoint has " + saved.size() + " tensors, expected " + dst.size());
            }
            for (int i = 0; i < dst.size(); i++) {
                dst.get(i).set(FloatBuffer.wrap(saved.get(i).toFloatArray()));
            }
        }
        return model;
    }
}
