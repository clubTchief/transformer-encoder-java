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
}
