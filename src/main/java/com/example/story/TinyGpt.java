package com.example.story;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

/**
 * Decoder-only (GPT-style) transformer built from the same ingredients as
 * ModernizedHierarchicalEncoderPipeline: Pre-LN, RoPE, SwiGLU. Three things change:
 *   1. CAUSAL MASK in attention  (encoder: every token sees every token; generator: only the past)
 *   2. TRAINABLE parameters       (requiresGradient=true, gamma/beta learnable, everything differentiable)
 *   3. LM HEAD tied to the token embedding, producing next-token logits
 * Batched: [B, T] token ids in, [B, T, V] logits out.
 *
 * Memory note (DJL): every array an op returns is owned by the manager of its left operand, so all
 * intermediates are rooted in the per-step {@code scope} manager that callers close each step.
 */
public final class TinyGpt {

    private record Layer(NDArray g1, NDArray b1, NDArray wq, NDArray wk, NDArray wv, NDArray wo,
                         NDArray g2, NDArray b2, NDArray wGate, NDArray wUp, NDArray wDown) {}

    private final GptConfig cfg;
    private final NDManager manager;
    private final NDArray tokEmb; // [V, d], also used (transposed) as the output head
    private final List<Layer> layers = new ArrayList<>();
    private final NDArray finalGamma;
    private final NDArray finalBeta;
    private final float[][] ropeCos; // [ctx, dk/2]
    private final float[][] ropeSin;

    public TinyGpt(NDManager manager, GptConfig cfg, long seed) {
        this.manager = manager;
        this.cfg = cfg;
        Engine.getInstance().setRandomSeed((int) seed);

        int d = cfg.dModel();
        int ff = cfg.dFF();
        float std = 0.02f;
        float resStd = std / (float) Math.sqrt(2.0 * cfg.numLayers()); // GPT-2 style residual scaling

        tokEmb = param(new Shape(cfg.vocabSize(), d), std);
        for (int l = 0; l < cfg.numLayers(); l++) {
            layers.add(new Layer(
                    ones(d), zeros(d),
                    param(new Shape(d, d), std), param(new Shape(d, d), std),
                    param(new Shape(d, d), std), param(new Shape(d, d), resStd),
                    ones(d), zeros(d),
                    param(new Shape(d, ff), std), param(new Shape(d, ff), std),
                    param(new Shape(ff, d), resStd)));
        }
        finalGamma = ones(d);
        finalBeta = zeros(d);

        int half = cfg.headDim() / 2;
        ropeCos = new float[cfg.contextLength()][half];
        ropeSin = new float[cfg.contextLength()][half];
        for (int pos = 0; pos < cfg.contextLength(); pos++) {
            for (int i = 0; i < half; i++) {
                double theta = pos / Math.pow(10000.0, (2.0 * i) / cfg.headDim()); // same formula as the repo
                ropeCos[pos][i] = (float) Math.cos(theta);
                ropeSin[pos][i] = (float) Math.sin(theta);
            }
        }
    }

    public GptConfig config() {
        return cfg;
    }

    /** All trainable tensors, in a fixed order (used by the optimizer and by checkpoints). */
    public List<NDArray> params() {
        List<NDArray> p = new ArrayList<>();
        p.add(tokEmb);
        for (Layer L : layers) {
            p.addAll(List.of(L.g1, L.b1, L.wq, L.wk, L.wv, L.wo, L.g2, L.b2, L.wGate, L.wUp, L.wDown));
        }
        p.add(finalGamma);
        p.add(finalBeta);
        return p;
    }

    /** ids: INT32 [B, T]  ->  logits: FLOAT32 [B, T, V]. */
    public NDArray forward(NDManager scope, NDArray ids) {
        long B = ids.getShape().get(0);
        long T = ids.getShape().get(1);
        if (T > cfg.contextLength()) {
            throw new IllegalArgumentException("T=" + T + " exceeds context length " + cfg.contextLength());
        }
        int d = cfg.dModel();
        int h = cfg.numHeads();
        int dk = cfg.headDim();
        int t = (int) T;

        NDArray cos = scope.create(Arrays.copyOfRange(ropeCos, 0, t));
        NDArray sin = scope.create(Arrays.copyOfRange(ropeSin, 0, t));
        NDArray mask = scope.create(causalMask(t));
        NDArray embT = tokEmb.transpose();
        embT.attach(scope);

        // Embedding lookup as one-hot @ table: differentiable w.r.t. the table, trivial for V ~ 100.
        NDArray oneHot = ids.reshape(B * T).toType(DataType.INT64, false).oneHot(cfg.vocabSize());
        NDArray x = oneHot.matMul(tokEmb).reshape(B, T, d);

        for (Layer L : layers) {
            // ---- sublayer 1: Pre-LN + causal multi-head attention with RoPE ----
            NDArray n1 = layerNorm(x, L.g1, L.b1);
            NDArray q = rope(splitHeads(n1.matMul(L.wq), B, T), cos, sin);
            NDArray k = rope(splitHeads(n1.matMul(L.wk), B, T), cos, sin);
            NDArray v = splitHeads(n1.matMul(L.wv), B, T);

            NDArray scores = q.matMul(k.transpose(0, 1, 3, 2)).div((float) Math.sqrt(dk)).add(mask);
            NDArray attended = scores.softmax(-1).matMul(v);                    // [B, h, T, dk]
            NDArray merged = attended.transpose(0, 2, 1, 3).reshape(B, T, d);  // [B, T, d]
            x = x.add(merged.matMul(L.wo));

            // ---- sublayer 2: Pre-LN + SwiGLU feed-forward ----
            NDArray n2 = layerNorm(x, L.g2, L.b2);
            NDArray gate = swish(n2.matMul(L.wGate));
            x = x.add(gate.mul(n2.matMul(L.wUp)).matMul(L.wDown));
        }

        x = layerNorm(x, finalGamma, finalBeta);
        return x.matMul(embT); // [B, T, V] (weight tying)
    }

    /** Mean next-token cross-entropy. logits [B,T,V], targets INT32 [B,T]. */
    public NDArray loss(NDArray logits, NDArray targets) {
        int V = cfg.vocabSize();
        long n = logits.getShape().get(0) * logits.getShape().get(1);
        NDArray logProbs = logits.reshape(n, V).logSoftmax(-1);
        NDArray oneHot = targets.reshape(n).toType(DataType.INT64, false).oneHot(V);
        return logProbs.mul(oneHot).sum().neg().div((float) n);
    }

    // ------------------------------------------------------------------ building blocks

    private NDArray layerNorm(NDArray x, NDArray gamma, NDArray beta) {
        NDArray mean = x.mean(new int[]{-1}, true);
        NDArray centered = x.sub(mean);
        NDArray var = centered.square().mean(new int[]{-1}, true);
        return centered.div(var.add(cfg.eps()).sqrt()).mul(gamma).add(beta);
    }

    private static NDArray swish(NDArray x) {
        return x.mul(Activation.sigmoid(x));
    }

    private NDArray splitHeads(NDArray t, long B, long T) {
        return t.reshape(B, T, cfg.numHeads(), cfg.headDim()).transpose(0, 2, 1, 3);
    }

    /**
     * Differentiable RoPE ("rotate-half" pairing: channel i with i + dk/2, as in LLaMA/HF).
     * The repo's applyRoPE pulls data into Java arrays, which cuts the autograd graph, and pairs
     * (2i, 2i+1). Both conventions are equivalent up to a fixed channel permutation.
     */
    private NDArray rope(NDArray x, NDArray cos, NDArray sin) {
        int half = cfg.headDim() / 2;
        NDArray x1 = x.get(new NDIndex("..., :" + half));
        NDArray x2 = x.get(new NDIndex("..., " + half + ":"));
        NDArray r1 = x1.mul(cos).sub(x2.mul(sin));
        NDArray r2 = x1.mul(sin).add(x2.mul(cos));
        return NDArrays.concat(new NDList(r1, r2), -1);
    }

    private static float[][] causalMask(int t) {
        float[][] m = new float[t][t];
        for (int i = 0; i < t; i++) {
            for (int j = i + 1; j < t; j++) m[i][j] = -1e9f; // position i may not look at j > i
        }
        return m;
    }

    // ------------------------------------------------------------------ init helpers

    private NDArray param(Shape shape, float std) {
        NDArray a = manager.randomNormal(0f, std, shape, DataType.FLOAT32);
        a.setRequiresGradient(true);
        return a;
    }

    private NDArray ones(int n) {
        NDArray a = manager.ones(new Shape(n));
        a.setRequiresGradient(true);
        return a;
    }

    private NDArray zeros(int n) {
        NDArray a = manager.zeros(new Shape(n));
        a.setRequiresGradient(true);
        return a;
    }

    // ------------------------------------------------------------------ checkpoints

    public void save(Path dir) throws IOException {
        Files.createDirectories(dir);
        NDList list = new NDList(params());
        Files.write(dir.resolve("model.params"), list.encode());
        Properties p = new Properties();
        p.setProperty("vocabSize", String.valueOf(cfg.vocabSize()));
        p.setProperty("dModel", String.valueOf(cfg.dModel()));
        p.setProperty("numHeads", String.valueOf(cfg.numHeads()));
        p.setProperty("numLayers", String.valueOf(cfg.numLayers()));
        p.setProperty("dFF", String.valueOf(cfg.dFF()));
        p.setProperty("contextLength", String.valueOf(cfg.contextLength()));
        p.setProperty("eps", String.valueOf(cfg.eps()));
        try (Writer w = Files.newBufferedWriter(dir.resolve("model.cfg"))) {
            p.store(w, "TinyGpt config");
        }
    }

    public static TinyGpt load(NDManager manager, Path dir) throws IOException {
        Properties p = new Properties();
        try (Reader r = Files.newBufferedReader(dir.resolve("model.cfg"))) {
            p.load(r);
        }
        GptConfig cfg = new GptConfig(
                Integer.parseInt(p.getProperty("vocabSize")), Integer.parseInt(p.getProperty("dModel")),
                Integer.parseInt(p.getProperty("numHeads")), Integer.parseInt(p.getProperty("numLayers")),
                Integer.parseInt(p.getProperty("dFF")), Integer.parseInt(p.getProperty("contextLength")),
                Float.parseFloat(p.getProperty("eps")));
        TinyGpt model = new TinyGpt(manager, cfg, 0);
        try (NDList saved = NDList.decode(manager, Files.readAllBytes(dir.resolve("model.params")))) {
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
