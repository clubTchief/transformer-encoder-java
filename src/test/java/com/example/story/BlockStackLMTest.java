package com.example.story;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class BlockStackLMTest {
    private static final int V = 20;
    private static final GptConfig CFG = new GptConfig(V, 16, 2, 2, 32, 8, 1e-5f);

    @Test
    void outputShapeIsTimeByVocab() {
        try (NDManager m = NDManager.newBaseManager()) {
            BlockStackLM model = new BlockStackLM(m, CFG, 1);
            NDArray logits = model.forward(m, m.create(new int[]{1, 2, 3, 4, 5}));
            assertThat(logits.getShape().getShape()).containsExactly(5L, (long) V);
        }
    }

    @Test
    void changingFutureTokensNeverChangesEarlierLogits() {
        try (NDManager m = NDManager.newBaseManager()) {
            BlockStackLM model = new BlockStackLM(m, CFG, 1);
            float[] a = model.forward(m, m.create(new int[]{1, 2, 3, 4, 5, 6})).toFloatArray();
            float[] b = model.forward(m, m.create(new int[]{1, 2, 3, 9, 8, 7})).toFloatArray();
            for (int i = 0; i < 3 * V; i++) assertThat(a[i]).isCloseTo(b[i], within(1e-5f));
            boolean laterDiffers = false;
            for (int i = 3 * V; i < 6 * V; i++) laterDiffers |= Math.abs(a[i] - b[i]) > 1e-6f;
            assertThat(laterDiffers).isTrue();
        }
    }

    @Test
    void gradientReachesEveryParameterIncludingEmbeddingsAndFinalNorm() {
        try (NDManager m = NDManager.newBaseManager()) {
            BlockStackLM model = new BlockStackLM(m, CFG, 1);
            try (NDManager s = m.newSubManager();
                 GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                NDArray loss = model.loss(model.forward(s, s.create(new int[]{1, 2, 3, 1, 2, 3})),
                        s.create(new int[]{2, 3, 1, 2, 3, 1}));
                gc.backward(loss);
            }
            List<NDArray> params = model.params();
            for (int i = 0; i < params.size(); i++) {
                float abs = params.get(i).getGradient().abs().sum().getFloat();
                assertThat(Float.isFinite(abs)).as("param #%d is finite", i).isTrue();
                assertThat(abs).as("param #%d %s", i, params.get(i).getShape()).isGreaterThan(0f);
            }
        }
    }

    @Test
    void adamWZeroesGradientsAfterEachStep() {
        try (NDManager m = NDManager.newBaseManager()) {
            BlockStackLM model = new BlockStackLM(m, CFG, 1);
            AdamW opt = new AdamW(m, model.params());
            try (NDManager s = m.newSubManager()) {
                try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                    gc.backward(model.loss(model.forward(s, s.create(new int[]{1, 2, 3, 4})),
                            s.create(new int[]{2, 3, 4, 5})));
                }
                opt.step(1e-3f, 0f, 1f);
            }
            for (NDArray p : model.params()) {
                assertThat(p.getGradient().abs().sum().getFloat()).isEqualTo(0f);
            }
        }
    }

    @Test
    void canOverfitAShortRepeatingSequence() {
        try (NDManager m = NDManager.newBaseManager()) {
            BlockStackLM model = new BlockStackLM(m, CFG, 1);
            AdamW opt = new AdamW(m, model.params());
            int[] x = {1, 2, 3, 1, 2, 3, 1, 2};
            int[] y = {2, 3, 1, 2, 3, 1, 2, 3};
            float first = 0;
            float last = 0;
            for (int step = 0; step < 150; step++) {
                try (NDManager s = m.newSubManager()) {
                    try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                        NDArray loss = model.loss(model.forward(s, s.create(x)), s.create(y));
                        gc.backward(loss);
                        last = loss.getFloat();
                        if (step == 0) first = last;
                    }
                    opt.step(5e-3f, 0f, 1f);
                }
            }
            assertThat(last).isLessThan(first * 0.5f);
        }
    }

    @Test
    void checkpointRoundTripKeepsLogits(@TempDir Path dir) throws Exception {
        try (NDManager m = NDManager.newBaseManager()) {
            BlockStackLM a = new BlockStackLM(m, CFG, 1);
            a.save(dir);
            BlockStackLM b = BlockStackLM.load(m, dir);
            NDArray ids = m.create(new int[]{1, 2, 3, 4});
            float[] la = a.forward(m, ids).toFloatArray();
            float[] lb = b.forward(m, ids).toFloatArray();
            for (int i = 0; i < la.length; i++) assertThat(lb[i]).isCloseTo(la[i], within(1e-6f));
        }
    }

    @Test
    void generatorProducesTextStartingWithThePrompt() {
        CharTokenizer tok = CharTokenizer.fromText("abcabcabc", 1);
        GptConfig cfg = new GptConfig(tok.vocabSize(), 16, 2, 2, 32, 8, 1e-5f);
        try (NDManager m = NDManager.newBaseManager()) {
            BlockStackLM model = new BlockStackLM(m, cfg, 1);
            String out = model.generator(m, tok).generate("ab", 5, 0f, 0, 1f, 1, s -> {});
            assertThat(out).startsWith("ab");
            assertThat(out.length()).isBetween(2, 7);
        }
    }
}
