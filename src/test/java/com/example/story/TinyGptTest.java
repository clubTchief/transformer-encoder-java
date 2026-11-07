package com.example.story;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class TinyGptTest {
    private static final int V = 20;
    private static final GptConfig CFG = new GptConfig(V, 16, 2, 2, 32, 8, 1e-5f);

    @Test
    void outputShapeIsBatchByTimeByVocab() {
        try (NDManager m = NDManager.newBaseManager()) {
            TinyGpt model = new TinyGpt(m, CFG, 1);
            NDArray logits = model.forward(m, m.create(new int[][]{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}));
            assertThat(logits.getShape().getShape()).containsExactly(2L, 5L, (long) V);
        }
    }

    /** The defining difference from the encoder: the future must not leak into the past. */
    @Test
    void changingFutureTokensNeverChangesEarlierLogits() {
        try (NDManager m = NDManager.newBaseManager()) {
            TinyGpt model = new TinyGpt(m, CFG, 1);
            float[] a = model.forward(m, m.create(new int[][]{{1, 2, 3, 4, 5, 6}})).toFloatArray();
            float[] b = model.forward(m, m.create(new int[][]{{1, 2, 3, 9, 8, 7}})).toFloatArray();
            for (int i = 0; i < 3 * V; i++) assertThat(a[i]).isCloseTo(b[i], within(1e-5f));
            boolean laterDiffers = false;
            for (int i = 3 * V; i < 6 * V; i++) laterDiffers |= Math.abs(a[i] - b[i]) > 1e-6f;
            assertThat(laterDiffers).isTrue();
        }
    }

    @Test
    void canOverfitAShortRepeatingSequence() {
        try (NDManager m = NDManager.newBaseManager()) {
            TinyGpt model = new TinyGpt(m, CFG, 1);
            AdamW opt = new AdamW(m, model.params());
            int[][] x = {{1, 2, 3, 1, 2, 3, 1, 2}};
            int[][] y = {{2, 3, 1, 2, 3, 1, 2, 3}};
            float first = 0;
            float last = 0;
            for (int step = 0; step < 150; step++) {
                try (NDManager s = m.newSubManager()) {
                    NDArray xb = s.create(x);
                    NDArray yb = s.create(y);
                    try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                        NDArray loss = model.loss(model.forward(s, xb), yb);
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
    void samplerGreedyAndTopOneAgree() {
        float[] logits = new float[V];
        logits[7] = 5f;
        logits[3] = 1f;
        assertThat(StoryGenerator.sample(logits, 0f, 0, 1f, new Random(1))).isEqualTo(7);
        assertThat(StoryGenerator.sample(logits, 1f, 1, 1f, new Random(1))).isEqualTo(7);
    }

    @Test
    void samplerNeverEmitsUnknownToken() {
        float[] logits = new float[V];
        logits[CharTokenizer.UNK] = 100f;
        for (int i = 0; i < 50; i++) {
            assertThat(StoryGenerator.sample(logits, 1f, 0, 1f, new Random(i))).isNotEqualTo(CharTokenizer.UNK);
        }
    }

    @Test
    void checkpointRoundTripKeepsLogits(@TempDir Path dir) throws Exception {
        try (NDManager m = NDManager.newBaseManager()) {
            TinyGpt a = new TinyGpt(m, CFG, 1);
            a.save(dir);
            TinyGpt b = TinyGpt.load(m, dir);
            NDArray ids = m.create(new int[][]{{1, 2, 3, 4}});
            float[] la = a.forward(m, ids).toFloatArray();
            float[] lb = b.forward(m, ids).toFloatArray();
            for (int i = 0; i < la.length; i++) assertThat(lb[i]).isCloseTo(la[i], within(1e-6f));
        }
    }
}
