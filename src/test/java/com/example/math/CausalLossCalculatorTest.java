package com.example.math;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import com.transformer.CausalLossCalculator;
import com.transformer.GradientEngine;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CausalLossCalculatorTest {

    private static double logSoftmax(double[] logits, int index) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : logits) max = Math.max(max, v);
        double sum = 0;
        for (double v : logits) sum += Math.exp(v - max);
        return logits[index] - max - Math.log(sum);
    }

    @Test
    @DisplayName("Uniform logits give a loss of ln(vocabSize)")
    void testUniformLogits() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray logits = manager.zeros(new Shape(2, 3, 5), DataType.FLOAT32);
            NDArray targets = manager.create(new float[][]{{1, 2, 3}, {4, 0, 1}}); // the dataset stores ids as float32
            float loss = CausalLossCalculator.calculateLoss(manager, logits, targets, 99).getFloat();
            assertEquals(Math.log(5), loss, 1e-5);
        }
    }

    @Test
    @DisplayName("Matches a hand-computed cross-entropy")
    void testMatchesHandComputedCrossEntropy() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray logits = manager.create(new float[][]{{1, 2, 3}, {3, 1, 0}}).reshape(1, 2, 3);
            NDArray targets = manager.create(new int[][]{{2, 0}});
            double expected = -(logSoftmax(new double[]{1, 2, 3}, 2) + logSoftmax(new double[]{3, 1, 0}, 0)) / 2;
            assertEquals(expected, CausalLossCalculator.calculateLoss(manager, logits, targets, 99).getFloat(), 1e-5);
        }
    }

    @Test
    @DisplayName("Padding positions do not count, and an all-padding batch gives 0 instead of NaN")
    void testPaddingIsIgnored() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray logits = manager.create(new float[][]{{1, 2, 3}, {3, 1, 0}}).reshape(1, 2, 3);

            NDArray oneRealToken = manager.create(new int[][]{{2, 0}}); // pad id 0: only position 0 counts
            double expected = -logSoftmax(new double[]{1, 2, 3}, 2);
            assertEquals(expected, CausalLossCalculator.calculateLoss(manager, logits, oneRealToken, 0).getFloat(), 1e-5);

            NDArray allPad = manager.create(new int[][]{{0, 0}});
            assertEquals(0.0f, CausalLossCalculator.calculateLoss(manager, logits, allPad, 0).getFloat(), 1e-6f);
        }
    }

    @Test
    @DisplayName("Gradients reach the logits, and padded positions get none")
    void testGradientFlow() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray logits = manager.create(new float[][]{{1, 2, 3}, {3, 1, 0}}).reshape(1, 2, 3);
            logits.setRequiresGradient(true);
            NDArray targets = manager.create(new int[][]{{2, 0}}); // pad id 0: position 1 is padding

            try (GradientCollector gc = new GradientEngine(0.01f).createCollector()) {
                gc.backward(CausalLossCalculator.calculateLoss(manager, logits, targets, 0));
            }

            float[] g = logits.getGradient().toFloatArray();
            // position 0: softmax - onehot (loss averaged over 1 real token)
            double[] p = {Math.exp(logSoftmax(new double[]{1, 2, 3}, 0)), Math.exp(logSoftmax(new double[]{1, 2, 3}, 1)),
                    Math.exp(logSoftmax(new double[]{1, 2, 3}, 2))};
            assertEquals(p[0], g[0], 1e-5);
            assertEquals(p[1], g[1], 1e-5);
            assertEquals(p[2] - 1.0, g[2], 1e-5);
            // position 1 is padding: no gradient
            assertEquals(0.0f, g[3], 1e-7f);
            assertEquals(0.0f, g[4], 1e-7f);
            assertEquals(0.0f, g[5], 1e-7f);
        }
    }
}
