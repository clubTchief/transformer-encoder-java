package com.example.story;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Random;

/**
 * Trains the DecoderBlock-based model. Each optimizer step averages gradients over {@code accum}
 * sequences (run one after another, gradients accumulate across backward() calls).
 *
 * mvn exec:java -Dexec.mainClass=com.example.story.TrainBlockStackLM \
 *   -Dexec.args="corpus=data/stories.txt out=checkpoints-stack steps=10000 accum=8"
 */
public final class TrainBlockStackLM {

    public static void main(String[] args) throws Exception {
        Map<String, String> a = Cli.parse(args);
        Path corpus = Path.of(Cli.require(a, "corpus"));
        Path out = Path.of(a.getOrDefault("out", "checkpoints-stack"));
        String preset = a.getOrDefault("preset", "tiny");
        int steps = Integer.parseInt(a.getOrDefault("steps", "10000"));
        int accum = Integer.parseInt(a.getOrDefault("accum", "8"));
        float maxLr = Float.parseFloat(a.getOrDefault("lr", "2e-3"));
        int evalEvery = Integer.parseInt(a.getOrDefault("evalEvery", "500"));
        int maxChars = Integer.parseInt(a.getOrDefault("maxChars", "20000000"));
        long seed = Long.parseLong(a.getOrDefault("seed", "1337"));

        String text = TrainStoryModel.readHead(corpus, maxChars);
        CharTokenizer tok = CharTokenizer.fromText(text, 50);
        int[] data = tok.encode(text);
        int split = (int) (data.length * 0.99);

        GptConfig cfg = "small".equals(preset) ? GptConfig.small(tok.vocabSize()) : GptConfig.tiny(tok.vocabSize());
        int T = cfg.contextLength();
        System.out.printf("chars=%d vocab=%d preset=%s d=%d layers=%d heads=%d ctx=%d accum=%d%n",
                data.length, tok.vocabSize(), preset, cfg.dModel(), cfg.numLayers(), cfg.numHeads(), T, accum);

        Files.createDirectories(out);
        tok.save(out.resolve("vocab.txt"));
        Random rng = new Random(seed);

        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = new BlockStackLM(manager, cfg, seed);
            AdamW opt = new AdamW(manager, model.params());
            StoryGenerator gen = model.generator(manager, tok);
            int warmup = Math.min(200, steps / 10);
            int[][] xb = new int[accum][T];
            int[][] yb = new int[accum][T];
            double running = 0;
            int count = 0;

            for (int step = 1; step <= steps; step++) {
                TrainStoryModel.sampleBatch(data, 0, split, rng, xb, yb);
                float stepLoss = 0f;
                try (NDManager scope = manager.newSubManager()) {
                    for (int i = 0; i < accum; i++) {
                        NDArray x = scope.create(xb[i]);
                        NDArray y = scope.create(yb[i]);
                        try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                            NDArray l = model.loss(model.forward(scope, x), y).div((float) accum);
                            gc.backward(l); // gradients add up across the accum sequences
                            stepLoss += l.getFloat();
                        }
                    }
                    opt.step(TrainStoryModel.lrAt(step, steps, maxLr, warmup), 0.1f, 1.0f); // collector closed
                }
                running += stepLoss;
                count++;
                if (step % 50 == 0) {
                    System.out.printf("step %6d | train loss %.4f | lr %.2e%n",
                            step, running / count, TrainStoryModel.lrAt(step, steps, maxLr, warmup));
                    running = 0;
                    count = 0;
                }
                if (step % evalEvery == 0 || step == steps) {
                    float val = evaluate(manager, model, data, split, data.length, T, rng, 16);
                    System.out.printf("---- step %d | val loss %.4f ----%n", step, val);
                    gen.generate("Once upon a time", 300, 0.8f, 40, 0.95f, step, System.out::print);
                    System.out.println("\n----");
                    model.save(out);
                }
            }
        }
    }

    static float evaluate(NDManager manager, BlockStackLM model, int[] data, int from, int to,
                          int T, Random rng, int nSequences) {
        int[][] xb = new int[nSequences][T];
        int[][] yb = new int[nSequences][T];
        TrainStoryModel.sampleBatch(data, from, to, rng, xb, yb);
        double total = 0;
        for (int i = 0; i < nSequences; i++) {
            try (NDManager s = manager.newSubManager()) {
                total += model.loss(model.forward(s, s.create(xb[i])), s.create(yb[i])).getFloat();
            }
        }
        return (float) (total / nSequences);
    }
}
