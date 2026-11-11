package com.example.math;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import com.transformer.AdamOptimizer;
import com.transformer.GradientEngine;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class AdamOptimizerTest {

    private static float[] backwardOnce(GradientEngine engine, NDArray w) {
        try (GradientCollector gc = engine.createCollector()) {
            gc.backward(w.mul(2f).sum());
        }
        return w.getGradient().toFloatArray();
    }

    @Test
    @DisplayName("step() resets the gradients, so the next backward() does not add onto the previous one")
    void testStepResetsGradients() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray w = manager.ones(new Shape(3), DataType.FLOAT32);
            w.setRequiresGradient(true);
            GradientEngine engine = new GradientEngine(0f);
            AdamOptimizer optimizer = new AdamOptimizer(List.of(w), 0f); // lr 0 keeps w fixed, isolating the reset

            float[] first = backwardOnce(engine, w);
            optimizer.step();
            float[] second = backwardOnce(engine, w);

            assertArrayEquals(new float[]{2f, 2f, 2f}, first, 1e-6f);
            assertArrayEquals(first, second, 1e-6f); // would be {4, 4, 4} if step() did not reset the gradient
        }
    }

    @Test
    @DisplayName("Adam moves the weight against the gradient")
    void testStepMovesWeightDownhill() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray w = manager.ones(new Shape(3), DataType.FLOAT32);
            w.setRequiresGradient(true);
            GradientEngine engine = new GradientEngine(0f);
            AdamOptimizer optimizer = new AdamOptimizer(List.of(w), 0.1f);

            backwardOnce(engine, w);
            optimizer.step();

            for (float v : w.toFloatArray()) {
                assertTrue(v < 1.0f, "weight should have decreased because the gradient is positive, was " + v);
            }
        }
    }

    @Test
    @DisplayName("The global gradient norm is reported, and clipping does not break the update")
    void testGradNormIsReported() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray w = manager.ones(new Shape(3), DataType.FLOAT32);
            w.setRequiresGradient(true);
            AdamOptimizer optimizer = new AdamOptimizer(List.of(w), 0.1f);

            backwardOnce(new GradientEngine(0f), w); // gradient [2, 2, 2]: norm sqrt(12)
            optimizer.step(0.1f, 0f, 1.0f);

            assertEquals(Math.sqrt(12), optimizer.getLastGradNorm(), 1e-4);
            for (float v : w.toFloatArray()) {
                assertEquals(0.9f, v, 1e-3f); // a first Adam step moves every weight by about the learning rate
            }
        }
    }

    @Test
    @DisplayName("The learning rate given to step() overrides the one from the constructor")
    void testPerStepLearningRate() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray w = manager.ones(new Shape(3), DataType.FLOAT32);
            w.setRequiresGradient(true);
            AdamOptimizer optimizer = new AdamOptimizer(List.of(w), 0.001f);

            backwardOnce(new GradientEngine(0f), w);
            optimizer.step(0.5f, 0f, 0f);

            for (float v : w.toFloatArray()) {
                assertEquals(0.5f, v, 1e-3f);
            }
        }
    }

    @Test
    @DisplayName("Weight decay shrinks matrices but not vectors")
    void testWeightDecayOnlyForMatrices() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray matrix = manager.ones(new Shape(2, 2), DataType.FLOAT32);
            NDArray vector = manager.ones(new Shape(2), DataType.FLOAT32);
            matrix.setRequiresGradient(true);
            vector.setRequiresGradient(true);
            AdamOptimizer optimizer = new AdamOptimizer(List.of(matrix, vector), 0.1f);

            try (GradientCollector gc = new GradientEngine(0f).createCollector()) {
                gc.backward(matrix.mul(0f).sum().add(vector.mul(0f).sum())); // zero gradients: only the decay acts
            }
            optimizer.step(0.1f, 0.5f, 0f);

            for (float v : matrix.toFloatArray()) assertEquals(0.95f, v, 1e-5f);
            for (float v : vector.toFloatArray()) assertEquals(1.0f, v, 1e-5f);
        }
    }

    @Test
    @DisplayName("A NaN gradient skips the update instead of turning every weight into NaN")
    void testNonFiniteGradientIsSkipped() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray w = manager.ones(new Shape(3), DataType.FLOAT32);
            w.setRequiresGradient(true);
            AdamOptimizer optimizer = new AdamOptimizer(List.of(w), 0.1f);

            try (GradientCollector gc = new GradientEngine(0f).createCollector()) {
                gc.backward(w.mul(Float.NaN).sum()); // the gradient of this loss is NaN
            }
            assertFalse(optimizer.step(0.1f, 0f, 1.0f), "the step must be reported as skipped");
            assertArrayEquals(new float[]{1f, 1f, 1f}, w.toFloatArray(), 0f);
            assertEquals(1, optimizer.getSkippedSteps());
            assertEquals(0, optimizer.getStepCount(), "a skipped step does not count");

            // the NaN gradient was cleared, so the next healthy step works normally
            backwardOnce(new GradientEngine(0f), w);
            assertTrue(optimizer.step(0.1f, 0f, 1.0f));
            for (float v : w.toFloatArray()) {
                assertTrue(v < 1.0f && Float.isFinite(v), "weights should move after a healthy step, was " + v);
            }
        }
    }
}
