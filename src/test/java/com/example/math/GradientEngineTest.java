package com.example.math;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class GradientEngineTest {

    private static float[] backwardOnce(GradientEngine engine, NDArray w) {
        try (GradientCollector gc = engine.createCollector()) {
            gc.backward(w.mul(2f).sum());
        }
        return w.getGradient().toFloatArray();
    }

    @Test
    @DisplayName("step() zeroes gradients, so the next backward() does not accumulate onto the old one")
    void stepZeroesGradients() {
        try (NDManager m = NDManager.newBaseManager()) {
            NDArray w = m.ones(new Shape(3), DataType.FLOAT32);
            w.setRequiresGradient(true);
            GradientEngine engine = new GradientEngine(0f); // lr 0 keeps w fixed, isolating the zeroing

            float[] g1 = backwardOnce(engine, w);
            engine.step(List.of(w));
            float[] g2 = backwardOnce(engine, w);

            assertArrayEquals(new float[]{2f, 2f, 2f}, g1, 1e-6f);
            assertArrayEquals(g1, g2, 1e-6f); // would be {4,4,4} if step() did not zero
        }
    }

    @Test
    @DisplayName("SGD update moves the weight by lr * gradient")
    void stepAppliesLearningRate() {
        try (NDManager m = NDManager.newBaseManager()) {
            NDArray w = m.ones(new Shape(3), DataType.FLOAT32);
            w.setRequiresGradient(true);
            GradientEngine engine = new GradientEngine(0.1f);

            backwardOnce(engine, w);
            engine.step(List.of(w));

            assertArrayEquals(new float[]{0.8f, 0.8f, 0.8f}, w.toFloatArray(), 1e-6f);
        }
    }

    @Test
    @DisplayName("Repeated SGD steps lower the loss of a real DecoderBlock")
    void sgdReducesBlockLoss() {
        try (NDManager m = NDManager.newBaseManager()) {
            int n = 6, d = 16;
            DecoderBlock block = new DecoderBlock(m, d, 2, 32, 1e-5f);
            NDArray mask = CausalMaskGenerator.createCausalMask(m, n).squeeze(0).squeeze(0);
            NDArray[] rope = ModernizedHierarchicalEncoderPipeline.precomputeRoPEFrequencies(m, n, d / 2);
            NDArray x = m.randomNormal(0.0f, 1.0f, new Shape(n, d), DataType.FLOAT32);
            GradientEngine engine = new GradientEngine(1e-4f);

            float first = 0, last = 0;
            for (int step = 0; step < 5; step++) {
                try (GradientCollector gc = engine.createCollector()) {
                    NDArray loss = block.forward(x, mask, rope[0], rope[1]).square().sum();
                    gc.backward(loss);
                    last = loss.getFloat();
                    if (step == 0) first = last;
                }
                engine.step(block.params()); // outside the collector
            }
            assertTrue(last < first, "loss should drop: first=" + first + " last=" + last);
        }
    }
}
