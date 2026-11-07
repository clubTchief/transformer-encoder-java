package com.example.math;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class DecoderBlockTest {

    private static final int N = 6;
    private static final int D_MODEL = 16;
    private static final int HEADS = 2;
    private static final int D_FF = 32;

    private NDManager manager;
    private DecoderBlock block;
    private NDArray mask;
    private NDArray[] rope;

    @BeforeEach
    void setUp() {
        manager = NDManager.newBaseManager();
        block = new DecoderBlock(manager, D_MODEL, HEADS, D_FF, 1e-5f);
        mask = CausalMaskGenerator.createCausalMask(manager, N).squeeze(0).squeeze(0);
        rope = ModernizedHierarchicalEncoderPipeline.buildRoPETables(manager, N, D_MODEL / HEADS);
    }

    @AfterEach
    void tearDown() {
        manager.close();
    }

    private NDArray randomInput() {
        return manager.randomNormal(0.0f, 1.0f, new Shape(N, D_MODEL), DataType.FLOAT32);
    }

    @Test
    @DisplayName("Forward keeps the [seqLen, dModel] shape")
    void testOutputShape() {
        NDArray out = block.apply(randomInput(), mask, rope[0], rope[1]);
        assertEquals(new Shape(N, D_MODEL), out.getShape());
    }

    @Test
    @DisplayName("Every parameter receives a finite, non-zero gradient (autograd graph is unbroken)")
    void testGradientReachesEveryParameter() {
        GradientEngine engine = new GradientEngine(0.01f);
        NDArray x = randomInput();

        try (GradientCollector collector = engine.createCollector()) {
            NDArray out = block.apply(x, mask, rope[0], rope[1]);
            NDArray loss = out.square().sum();
            collector.backward(loss);
        }

        List<NDArray> params = block.params();
        assertEquals(14, params.size());
        for (int i = 0; i < params.size(); i++) {
            float absSum;
            try {
                absSum = params.get(i).getGradient().abs().sum().getFloat();
            } catch (Exception e) {
                fail("Parameter #" + i + " " + params.get(i).getShape() + " has no gradient: " + e);
                return;
            }
            assertTrue(Float.isFinite(absSum), "Parameter #" + i + " gradient is NaN/Inf");
            assertTrue(absSum > 0f, "Parameter #" + i + " " + params.get(i).getShape()
                    + " gradient is all zeros -> graph is broken before this tensor");
        }
    }

    @Test
    @DisplayName("Causality: changing the last token never changes earlier positions")
    void testCausality() {
        float[][] base = new float[N][D_MODEL];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < D_MODEL; j++) base[i][j] = (float) Math.sin(i * 31 + j);
        }
        float[][] changed = new float[N][];
        for (int i = 0; i < N; i++) changed[i] = base[i].clone();
        for (int j = 0; j < D_MODEL; j++) changed[N - 1][j] += 5.0f; // perturb the LAST token only

        float[] a = block.apply(manager.create(base), mask, rope[0], rope[1]).toFloatArray();
        float[] b = block.apply(manager.create(changed), mask, rope[0], rope[1]).toFloatArray();

        for (int i = 0; i < (N - 1) * D_MODEL; i++) {
            assertEquals(a[i], b[i], 1e-5f, "Earlier position changed at flat index " + i);
        }
        boolean lastDiffers = false;
        for (int i = (N - 1) * D_MODEL; i < N * D_MODEL; i++) lastDiffers |= Math.abs(a[i] - b[i]) > 1e-5f;
        assertTrue(lastDiffers, "The perturbed token itself should change");
    }

    @Test
    @DisplayName("Parameters persist: two forward passes on the same input give identical output")
    void testParametersPersistAcrossForwardPasses() {
        NDArray x = randomInput();
        float[] first = block.apply(x, mask, rope[0], rope[1]).toFloatArray();
        float[] second = block.apply(x, mask, rope[0], rope[1]).toFloatArray();
        assertArrayEquals(first, second, 1e-6f);
    }

    @Test
    @DisplayName("One SGD step on the block lowers the loss")
    void testSgdStepReducesLoss() {
        GradientEngine engine = new GradientEngine(1e-4f);
        NDArray x = randomInput();

        float before;
        try (GradientCollector collector = engine.createCollector()) {
            NDArray loss = block.apply(x, mask, rope[0], rope[1]).square().sum();
            before = loss.getFloat();
            collector.backward(loss);
        }
        engine.step(block.params()); // outside the collector

        float after = block.apply(x, mask, rope[0], rope[1]).square().sum().getFloat();
        assertTrue(after < before, "loss should drop: before=" + before + " after=" + after);
    }
}
