package com.example.story;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;
import java.util.stream.IntStream;

/** Autoregressive sampling: feed the prompt, predict one char, append, repeat. */
public final class StoryGenerator {
    /** Next-token logits [V] for the LAST position of {@code window} (token ids, oldest first). */
    @FunctionalInterface
    public interface NextTokenLogits {
        float[] logitsFor(NDManager scope, int[] window);
    }

    private final NDManager manager;
    private final int contextLength;
    private final NextTokenLogits logits;
    private final CharTokenizer tok;

    public StoryGenerator(NDManager manager, TinyGpt model, CharTokenizer tok) {
        this(manager, model.config().contextLength(), (scope, window) -> {
            NDArray out = model.forward(scope, scope.create(new int[][]{window})); // [1, T, V]
            return out.get(new NDIndex("0, " + (window.length - 1) + ", :")).toFloatArray();
        }, tok);
    }

    public StoryGenerator(NDManager manager, int contextLength, NextTokenLogits logits, CharTokenizer tok) {
        this.manager = manager;
        this.contextLength = contextLength;
        this.logits = logits;
        this.tok = tok;
    }

    public String generate(String prompt, int maxNewTokens, float temperature, int topK, float topP,
                           long seed, Consumer<String> sink) {
        Random rng = new Random(seed);
        List<Integer> ids = new ArrayList<>();
        ids.add(CharTokenizer.EOS); // every training story starts right after an EOS
        for (int id : tok.encode(prompt)) ids.add(id);

        StringBuilder out = new StringBuilder(prompt);
        sink.accept(prompt);

        for (int n = 0; n < maxNewTokens; n++) {
            int from = Math.max(0, ids.size() - contextLength); // sliding window
            int T = ids.size() - from;
            int[] window = new int[T];
            for (int i = 0; i < T; i++) window[i] = ids.get(from + i);

            int next;
            try (NDManager scope = manager.newSubManager()) {
                float[] last = logits.logitsFor(scope, window);
                next = sample(last, temperature, topK, topP, rng);
            }
            if (next == CharTokenizer.EOS) break; // the model decided the story is over
            ids.add(next);
            String piece = tok.decodeToken(next);
            out.append(piece);
            sink.accept(piece);
        }
        return out.toString();
    }

    /** temperature <= 0 -> greedy. topK <= 0 disables top-k. topP >= 1 disables nucleus filtering. */
    static int sample(float[] logits, float temperature, int topK, float topP, Random rng) {
        float[] l = logits.clone();
        l[CharTokenizer.UNK] = Float.NEGATIVE_INFINITY;
        if (temperature <= 0f) return argmax(l);

        int V = l.length;
        double max = Double.NEGATIVE_INFINITY;
        for (float x : l) max = Math.max(max, x);
        final double[] p = new double[V];
        double sum = 0;
        for (int i = 0; i < V; i++) {
            p[i] = Math.exp((l[i] - max) / temperature);
            sum += p[i];
        }
        Integer[] order = IntStream.range(0, V).boxed().toArray(Integer[]::new);
        Arrays.sort(order, (a, b) -> Double.compare(p[b], p[a]));

        int keep = (topK > 0) ? Math.min(V, topK) : V;
        double kept = 0;
        int n = 0;
        while (n < keep) {
            kept += p[order[n]] / sum;
            n++;
            if (kept >= topP) break;
        }
        double r = rng.nextDouble() * kept;
        double acc = 0;
        for (int i = 0; i < n; i++) {
            acc += p[order[i]] / sum;
            if (r <= acc) return order[i];
        }
        return order[n - 1];
    }

    static int argmax(float[] a) {
        int best = 0;
        for (int i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
        return best;
    }
}
