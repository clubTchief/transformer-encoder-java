package com.dictionary;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;
import com.example.story.CharTokenizer;
import com.generation.Sampler;
import com.generation.TextGenerator;
import com.transformer.AdamOptimizer;
import com.transformer.BlockStackLM;
import com.transformer.CausalLossCalculator;
import com.transformer.CausalMaskGenerator;
import com.transformer.LMConfig;
import com.transformer.ModelCheckpoint;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Trains the repo's own decoder stack (BlockStackLM) on a text file, character by character, and writes a
 * self-describing checkpoint that CorpusGeneratorCLI can use without knowing anything about the training run.
 *
 * mvn -q compile exec:java -Dexec.mainClass=com.dictionary.CorpusTrainer -Dexec.args="corpus=data/stories.txt steps=3000"
 */
public final class CorpusTrainer {

    /** Everything that can be set from the command line; the defaults are a ~0.8M parameter model that fits a laptop CPU. */
    public static final class Settings {
        public Path corpus;
        public Path checkpoint = Path.of("models/tinystories.ckpt");
        public int maxChars = 20_000_000;
        public int minCharFreq = 50;
        public double validationFraction = 0.01;

        public int numLayers = 4;
        public int dModel = 128;
        public int numHeads = 4;
        public int dFF = 352;
        public int seqLen = 128;

        public int batchSize = 32;
        public int steps = 5000;
        public int warmupSteps = -1; // -1: min(200, steps / 10)
        public int evalEvery = 500;
        public int evalBatches = 8;
        public int sampleTokens = 200; // 0 turns the sample text at every evaluation off
        public float learningRate = 2e-3f;
        public float weightDecay = 0.1f;
        public float gradClip = 1.0f;
        public long seed = 1337;

        private static final Set<String> KNOWN = Set.of("corpus", "out", "maxChars", "minFreq", "valFraction", "layers",
                "dModel", "heads", "dFF", "seqLen", "batch", "steps", "warmup", "evalEvery", "evalBatches", "sampleTokens",
                "lr", "wd", "clip", "seed");

        public static Settings fromArgs(String[] args) {
            Map<String, String> a = Args.parse(args);
            Args.rejectUnknown(a, KNOWN);
            Settings s = new Settings();
            if (a.containsKey("corpus")) s.corpus = Path.of(a.get("corpus"));
            if (a.containsKey("out")) s.checkpoint = Path.of(a.get("out"));
            if (a.containsKey("maxChars")) s.maxChars = Integer.parseInt(a.get("maxChars"));
            if (a.containsKey("minFreq")) s.minCharFreq = Integer.parseInt(a.get("minFreq"));
            if (a.containsKey("valFraction")) s.validationFraction = Double.parseDouble(a.get("valFraction"));
            if (a.containsKey("layers")) s.numLayers = Integer.parseInt(a.get("layers"));
            if (a.containsKey("dModel")) s.dModel = Integer.parseInt(a.get("dModel"));
            if (a.containsKey("heads")) s.numHeads = Integer.parseInt(a.get("heads"));
            if (a.containsKey("dFF")) s.dFF = Integer.parseInt(a.get("dFF"));
            if (a.containsKey("seqLen")) s.seqLen = Integer.parseInt(a.get("seqLen"));
            if (a.containsKey("batch")) s.batchSize = Integer.parseInt(a.get("batch"));
            if (a.containsKey("steps")) s.steps = Integer.parseInt(a.get("steps"));
            if (a.containsKey("warmup")) s.warmupSteps = Integer.parseInt(a.get("warmup"));
            if (a.containsKey("evalEvery")) s.evalEvery = Integer.parseInt(a.get("evalEvery"));
            if (a.containsKey("evalBatches")) s.evalBatches = Integer.parseInt(a.get("evalBatches"));
            if (a.containsKey("sampleTokens")) s.sampleTokens = Integer.parseInt(a.get("sampleTokens"));
            if (a.containsKey("lr")) s.learningRate = Float.parseFloat(a.get("lr"));
            if (a.containsKey("wd")) s.weightDecay = Float.parseFloat(a.get("wd"));
            if (a.containsKey("clip")) s.gradClip = Float.parseFloat(a.get("clip"));
            if (a.containsKey("seed")) s.seed = Long.parseLong(a.get("seed"));
            return s;
        }
    }

    /** Mean training loss of the first and of the last 10 steps, and the last validation loss. */
    public record Result(float firstLoss, float lastLoss, float validationLoss, int steps) {
    }

    private CorpusTrainer() {
    }

    public static void main(String[] args) throws IOException {
        Settings settings = Settings.fromArgs(args);
        if (settings.corpus == null) {
            throw new IllegalArgumentException("Missing corpus=<text file>. Example: corpus=data/stories.txt");
        }
        train(settings, System.out);
    }

    public static Result train(Settings s, PrintStream out) throws IOException {
        out.println("Reading " + s.corpus + " ...");
        return train(CharCorpus.load(s.corpus, s.maxChars, s.minCharFreq, s.validationFraction), s, out);
    }

    public static Result train(CharCorpus corpus, Settings s, PrintStream out) throws IOException {
        int window = s.seqLen;
        CharTokenizer tokenizer = corpus.tokenizer();
        LMConfig config = new LMConfig(s.numLayers, s.dModel, s.numHeads, s.dFF, tokenizer.vocabSize(), 1e-5f);
        int warmup = s.warmupSteps >= 0 ? s.warmupSteps : Math.min(200, s.steps / 10);

        out.printf("tokens: %d train / %d validation | vocab %d | layers %d, dModel %d, heads %d, dFF %d | window %d, batch %d%n",
                corpus.trainTokens(), corpus.validationTokens(), tokenizer.vocabSize(),
                config.numLayers(), config.dModel(), config.numHeads(), config.dFF(), window, s.batchSize);
        if (!corpus.hasHeldOutValidation(window)) {
            out.println("note: no held-out text for validation (corpus too small), so the validation loss is measured on training text");
        }

        Random rng = new Random(s.seed);
        int[][] inputs = new int[s.batchSize][window];
        int[][] targets = new int[s.batchSize][window];
        float[] losses = new float[s.steps];
        float validationLoss = Float.NaN;
        double windowSum = 0;
        int windowCount = 0;

        try (NDManager manager = NDManager.newBaseManager()) {
            Engine.getInstance().setRandomSeed((int) s.seed);
            BlockStackLM model = new BlockStackLM(manager, config);
            AdamOptimizer optimizer = new AdamOptimizer(model.params(), s.learningRate, 0.9f, 0.95f, 1e-8f);
            NDArray mask = CausalMaskGenerator.createCausalMask(manager, window).squeeze(0).squeeze(0);

            for (int step = 1; step <= s.steps; step++) {
                corpus.sampleTrainBatch(rng, inputs, targets);
                float lr = lrAt(step, s.steps, s.learningRate, warmup);

                float loss;
                try (NDManager scope = manager.newSubManager()) {
                    NDArray inputIds = scope.create(inputs);
                    NDArray targetIds = scope.create(targets);
                    try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                        NDArray logits = model.forwardIds(scope, inputIds, mask);
                        NDArray lossTensor = CausalLossCalculator.calculateLoss(scope, logits, targetIds, -1); // no padding
                        gc.backward(lossTensor);
                        loss = lossTensor.getFloat();
                    }
                    optimizer.step(lr, s.weightDecay, s.gradClip); // collector closed: gradient mode is off
                }

                if (!Float.isFinite(loss)) {
                    throw new IllegalStateException("The loss became " + loss + " at step " + step + ". Lower the learning rate (lr=...).");
                }
                losses[step - 1] = loss;
                windowSum += loss;
                windowCount++;
                if (step % 50 == 0) {
                    out.printf("step %6d | train loss %.4f | lr %.2e | grad norm %.2f%n",
                            step, windowSum / windowCount, lr, optimizer.getLastGradNorm());
                    windowSum = 0;
                    windowCount = 0;
                }

                if (step % s.evalEvery == 0 || step == s.steps) {
                    validationLoss = validationLoss(manager, model, corpus, mask, s.batchSize, window, s.evalBatches);
                    out.printf("---- step %d | validation loss %.4f ----%n", step, validationLoss);
                    if (s.sampleTokens > 0) {
                        out.println(sample(manager, model, tokenizer, window, s.sampleTokens, step));
                        out.println("----");
                    }
                    ModelCheckpoint.save(s.checkpoint, model, tokenizer.chars(), metadata(s, step, validationLoss));
                }
            }
        }

        int edge = Math.min(10, s.steps);
        return new Result(mean(losses, 0, edge), mean(losses, s.steps - edge, s.steps), validationLoss, s.steps);
    }

    static float lrAt(int step, int total, float max, int warmup) {
        if (step < warmup) {
            return max * (step + 1) / warmup;
        }
        double progress = (step - warmup) / (double) Math.max(1, total - warmup);
        return (float) (0.1 * max + 0.9 * max * 0.5 * (1 + Math.cos(Math.PI * progress)));
    }

    private static float validationLoss(NDManager manager, BlockStackLM model, CharCorpus corpus, NDArray mask,
                                        int batchSize, int window, int batches) {
        Random rng = new Random(12345); // the same windows at every evaluation, so the numbers can be compared
        int[][] inputs = new int[batchSize][window];
        int[][] targets = new int[batchSize][window];
        double total = 0;
        for (int i = 0; i < batches; i++) {
            corpus.sampleValidationBatch(rng, inputs, targets);
            try (NDManager scope = manager.newSubManager()) {
                NDArray logits = model.forwardIds(scope, scope.create(inputs), mask);
                total += CausalLossCalculator.calculateLoss(scope, logits, scope.create(targets), -1).getFloat();
            }
        }
        return (float) (total / batches);
    }

    private static String sample(NDManager manager, BlockStackLM model, CharTokenizer tokenizer, int window, int tokens, long seed) {
        StringBuilder text = new StringBuilder();
        new TextGenerator(model, new Sampler(seed)).generate(manager, new int[]{CharTokenizer.EOS}, tokens,
                CharTokenizer.EOS, 0.8f, 40, 0.95f, 1.0f, window, id -> text.append(tokenizer.decodeToken(id)));
        return text.toString();
    }

    private static Map<String, String> metadata(Settings s, int step, float validationLoss) {
        Map<String, String> meta = new LinkedHashMap<>();
        meta.put("seqLen", String.valueOf(s.seqLen));
        meta.put("steps", String.valueOf(step));
        meta.put("validationLoss", String.valueOf(validationLoss));
        meta.put("corpus", s.corpus == null ? "(in memory)" : s.corpus.toString());
        meta.put("seed", String.valueOf(s.seed));
        meta.put("trainedAt", Instant.now().toString());
        return meta;
    }

    private static float mean(float[] values, int from, int to) {
        double sum = 0;
        for (int i = from; i < to; i++) {
            sum += values[i];
        }
        return (float) (sum / Math.max(1, to - from));
    }
}
