package com.example.story;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Random;

/**
 * mvn exec:java -Dexec.mainClass=com.example.story.TrainStoryModel \
 *   -Dexec.args="corpus=data/TinyStoriesV2-GPT4-valid.txt out=checkpoints steps=10000"
 */
public final class TrainStoryModel {

    public static void main(String[] args) throws Exception {
        Map<String, String> a = Cli.parse(args);
        Path corpus = Path.of(Cli.require(a, "corpus"));
        Path out = Path.of(a.getOrDefault("out", "checkpoints"));
        String preset = a.getOrDefault("preset", "tiny");
        int steps = Integer.parseInt(a.getOrDefault("steps", "10000"));
        int batch = Integer.parseInt(a.getOrDefault("batch", "32"));
        float maxLr = Float.parseFloat(a.getOrDefault("lr", "2e-3"));
        int evalEvery = Integer.parseInt(a.getOrDefault("evalEvery", "500"));
        int maxChars = Integer.parseInt(a.getOrDefault("maxChars", "20000000"));
        long seed = Long.parseLong(a.getOrDefault("seed", "1337"));

        String text = readHead(corpus, maxChars);
        CharTokenizer tok = CharTokenizer.fromText(text, 50);
        int[] data = tok.encode(text);
        int split = (int) (data.length * 0.99);

        GptConfig cfg = "small".equals(preset) ? GptConfig.small(tok.vocabSize()) : GptConfig.tiny(tok.vocabSize());
        int T = cfg.contextLength();
        System.out.printf("chars=%d vocab=%d preset=%s d=%d layers=%d heads=%d ctx=%d%n",
                data.length, tok.vocabSize(), preset, cfg.dModel(), cfg.numLayers(), cfg.numHeads(), T);

        Files.createDirectories(out);
        tok.save(out.resolve("vocab.txt"));
        Random rng = new Random(seed);

        try (NDManager manager = NDManager.newBaseManager()) {
            TinyGpt model = new TinyGpt(manager, cfg, seed);
            AdamW opt = new AdamW(manager, model.params());
            StoryGenerator gen = new StoryGenerator(manager, model, tok);
            int warmup = Math.min(200, steps / 10);
            int[][] xb = new int[batch][T];
            int[][] yb = new int[batch][T];
            double running = 0;
            int count = 0;

            for (int step = 1; step <= steps; step++) {
                sampleBatch(data, 0, split, rng, xb, yb);
                float loss;
                try (NDManager scope = manager.newSubManager()) {
                    NDArray x = scope.create(xb);
                    NDArray y = scope.create(yb);
                    try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                        NDArray l = model.loss(model.forward(scope, x), y);
                        gc.backward(l);
                        loss = l.getFloat();
                    }
                    opt.step(lrAt(step, steps, maxLr, warmup), 0.1f, 1.0f); // collector closed: grad mode off
                }
                running += loss;
                count++;
                if (step % 50 == 0) {
                    System.out.printf("step %6d | train loss %.4f | lr %.2e%n",
                            step, running / count, lrAt(step, steps, maxLr, warmup));
                    running = 0;
                    count = 0;
                }
                if (step % evalEvery == 0 || step == steps) {
                    float val = evaluate(manager, model, data, split, data.length, batch, T, rng, 20);
                    System.out.printf("---- step %d | val loss %.4f ----%n", step, val);
                    gen.generate("Once upon a time", 300, 0.8f, 40, 0.95f, step, System.out::print);
                    System.out.println("\n----");
                    model.save(out);
                }
            }
        }
    }

    static void sampleBatch(int[] data, int from, int to, Random rng, int[][] x, int[][] y) {
        int T = x[0].length;
        for (int b = 0; b < x.length; b++) {
            int off = from + rng.nextInt(to - from - T - 1);
            for (int t = 0; t < T; t++) {
                x[b][t] = data[off + t];
                y[b][t] = data[off + t + 1]; // target = next character
            }
        }
    }

    static float evaluate(NDManager manager, TinyGpt model, int[] data, int from, int to,
                          int batch, int T, Random rng, int nBatches) {
        int[][] xb = new int[batch][T];
        int[][] yb = new int[batch][T];
        double total = 0;
        for (int i = 0; i < nBatches; i++) {
            sampleBatch(data, from, to, rng, xb, yb);
            try (NDManager s = manager.newSubManager()) {
                total += model.loss(model.forward(s, s.create(xb)), s.create(yb)).getFloat();
            }
        }
        return (float) (total / nBatches);
    }

    /** Linear warmup, then cosine decay to 10% of the peak. */
    static float lrAt(int step, int total, float max, int warmup) {
        if (step < warmup) return max * (step + 1) / warmup;
        double progress = (step - warmup) / (double) Math.max(1, total - warmup);
        return (float) (0.1 * max + 0.9 * max * 0.5 * (1 + Math.cos(Math.PI * progress)));
    }

    static String readHead(Path file, int maxChars) throws Exception {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader r = Files.newBufferedReader(file)) {
            char[] buf = new char[1 << 16];
            int n;
            while (sb.length() < maxChars && (n = r.read(buf)) > 0) sb.append(buf, 0, n);
        }
        return sb.toString();
    }
}
