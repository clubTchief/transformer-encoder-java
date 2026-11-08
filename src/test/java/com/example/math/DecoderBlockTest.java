package com.example.math;

import com.decoder.DecoderBlock;
import com.decoder.DecoderBlockResult;
import com.transformer.GradientEngine;
import com.transformer.CausalMaskGenerator;
import com.encoder.ModernizedHierarchicalEncoderPipeline;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class DecoderBlockTest {

    private NDManager manager;

    @BeforeEach
    void setUp() {
        manager = NDManager.newBaseManager();
    }

    @AfterEach
    void tearDown() {
        manager.close();
    }

    @Test
    @DisplayName("Verify DecoderBlock forward pass executes cleanly and preserves shape")
    void testForwardPassShapePreservation() {
        int seqLen = 4;
        int dModel = 384;
        int numHeads = 8;
        int dFF = 4 * dModel;
        int dK = dModel / numHeads;
        float eps = 1e-5f;

        // 1. Create DecoderBlock instance
        DecoderBlock block = new DecoderBlock(manager, dModel, numHeads, dFF, eps);

        // 2. Prepare dummy input X [seqLen, dModel]
        NDArray X = manager.randomNormal(0.0f, 1.0f, new Shape(seqLen, dModel), DataType.FLOAT32);

        // 3. Prepare 2D causal mask [seqLen, seqLen]
        NDArray causalMask4D = CausalMaskGenerator.createCausalMask(manager, seqLen);
        NDArray causalMask = causalMask4D.squeeze(0).squeeze(0);

        // 4. Pre-compute RoPE tables using correct pipeline method
        NDArray[] ropeTables = ModernizedHierarchicalEncoderPipeline.precomputeRoPEFrequencies(manager, seqLen, dK);
        NDArray cos = ropeTables[0];
        NDArray sin = ropeTables[1];

        // 5. Run forward pass
        NDArray output = block.forward(X, causalMask, cos, sin);

        assertNotNull(output);
        assertEquals(new Shape(seqLen, dModel), output.getShape(), "DecoderBlock forward must preserve input tensor shape");
    }

    @Test
    @DisplayName("Verify DecoderBlock registers all 14 trainable parameters with gradient tracking")
    void testParameterRegistration() {
        int dModel = 384;
        int numHeads = 8;
        int dFF = 1536;
        float eps = 1e-5f;

        DecoderBlock block = new DecoderBlock(manager, dModel, numHeads, dFF, eps);
        List<NDArray> params = block.params();

        // 2 LayerNorms x 2 params (gamma, beta) + 4 MHA weights + 7 FFN weights/biases = 15 params
        assertEquals(15, params.size(), "DecoderBlock must export 15 trainable parameters");

        for (int i = 0; i < params.size(); i++) {
            NDArray param = params.get(i);
            assertNotNull(param, "Parameter at index " + i + " cannot be null");
            assertTrue(param.hasGradient(), "Parameter at index " + i + " must have gradients enabled");
        }
    }

    @Test
    @DisplayName("Verify autograd gradient flow reaches every parameter in DecoderBlock")
    public void testGradientFlowThroughBlock() {
        try (NDManager manager = NDManager.newBaseManager()) {
            DecoderBlock block = new DecoderBlock(manager, 64, 4, 256, 1e-5f);
            
            // 1. MUST set requiresGradient on all block parameters
            for (NDArray param : block.params()) {
                param.setRequiresGradient(true);
            }

            NDArray inputX = manager.randomUniform(-1f, 1f, new Shape(2, 12, 64), DataType.FLOAT32);
            inputX.setRequiresGradient(true); // Track gradients for input tensor

            try (GradientCollector gc = manager.getEngine().newGradientCollector()) {
                NDArray mask = CausalMaskGenerator.createCausalMask(manager, 12).squeeze(0).squeeze(0);
                NDArray output = block.forward(manager, inputX, mask);
                
                // 2. Reduce output to a scalar for backward pass
                NDArray loss = output.sum();
                
                // 3. Compute gradients across the tracked graph
                gc.backward(loss);
            }

            // 4. Assert gradients are populated and non-null
            for (NDArray param : block.params()) {
                NDArray grad = param.getGradient();
                Assertions.assertNotNull(grad, "Gradient should not be null for parameter");
            }
        }
    }

    @Test
    @DisplayName("Causality: changing the last token never changes earlier positions")
    void testCausality() {
        int n = 6, d = 16;
        DecoderBlock block = new DecoderBlock(manager, d, 2, 32, 1e-5f);
        NDArray mask = CausalMaskGenerator.createCausalMask(manager, n).squeeze(0).squeeze(0);
        NDArray[] rope = ModernizedHierarchicalEncoderPipeline.precomputeRoPEFrequencies(manager, n, d / 2);

        float[][] base = new float[n][d];
        for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) base[i][j] = (float) Math.sin(i * 31 + j);
        float[][] changed = new float[n][];
        for (int i = 0; i < n; i++) changed[i] = base[i].clone();
        for (int j = 0; j < d; j++) changed[n - 1][j] += 5.0f; // perturb ONLY the last token

        float[] a = block.forward(manager.create(base), mask, rope[0], rope[1]).toFloatArray();
        float[] b = block.forward(manager.create(changed), mask, rope[0], rope[1]).toFloatArray();

        for (int i = 0; i < (n - 1) * d; i++) {
            assertEquals(a[i], b[i], 1e-5f, "Earlier position changed at flat index " + i);
        }
        boolean lastDiffers = false;
        for (int i = (n - 1) * d; i < n * d; i++) lastDiffers |= Math.abs(a[i] - b[i]) > 1e-5f;
        assertTrue(lastDiffers, "The perturbed token itself should change");
    }

    @Test
    @DisplayName("Parameters persist: two forward passes on the same input give identical output")
    void testParametersPersistAcrossForwardPasses() {
        int n = 4, d = 16;
        DecoderBlock block = new DecoderBlock(manager, d, 2, 32, 1e-5f);
        NDArray mask = CausalMaskGenerator.createCausalMask(manager, n).squeeze(0).squeeze(0);
        NDArray[] rope = ModernizedHierarchicalEncoderPipeline.precomputeRoPEFrequencies(manager, n, d / 2);
        NDArray x = manager.randomNormal(0.0f, 1.0f, new Shape(n, d), DataType.FLOAT32);

        assertArrayEquals(block.forward(x, mask, rope[0], rope[1]).toFloatArray(),
                block.forward(x, mask, rope[0], rope[1]).toFloatArray(), 1e-6f);
    }

    // ------------------------------------------------------------------ mask contract

    @Test
    @DisplayName("A mask that does not match the attention scores is rejected, never silently ignored")
    void testWrongSizedMaskThrows() {
        int n = 6, d = 16;
        DecoderBlock block = new DecoderBlock(manager, d, 2, 32, 1e-5f);
        NDArray x = manager.randomNormal(0.0f, 1.0f, new Shape(n, d), DataType.FLOAT32);
        NDArray tooSmall = CausalMaskGenerator.createCausalMask(manager, n - 1).squeeze(0).squeeze(0);
        NDArray tooBig = CausalMaskGenerator.createCausalMask(manager, n + 1).squeeze(0).squeeze(0);

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> block.forward(manager, x, tooSmall));
        assertTrue(e.getMessage().contains("expected [6, 6]"), "message should say what shape was expected: " + e.getMessage());
        assertThrows(IllegalArgumentException.class, () -> block.forward(manager, x, tooBig));
    }

    @Test
    @DisplayName("No mask is rejected when several tokens are processed (it would silently be bidirectional)")
    void testMissingMaskRejectedForMultipleTokens() {
        DecoderBlock block = new DecoderBlock(manager, 16, 2, 32, 1e-5f);
        NDArray x = manager.randomNormal(0.0f, 1.0f, new Shape(4, 16), DataType.FLOAT32);
        assertThrows(IllegalArgumentException.class, () -> block.forward(manager, x, null));
    }

    @Test
    @DisplayName("No mask is fine for a single token, which can only attend to itself")
    void testMissingMaskAllowedForSingleToken() {
        DecoderBlock block = new DecoderBlock(manager, 16, 2, 32, 1e-5f);
        NDArray x = manager.randomNormal(0.0f, 1.0f, new Shape(1, 16), DataType.FLOAT32);
        assertEquals(new Shape(1, 16), block.forward(manager, x, null).getShape());
    }

    @Test
    @DisplayName("Cached decoding step needs a mask of exactly [newTokens, cached + new]")
    void testCachedStepRequiresExactMask() {
        int d = 16;
        DecoderBlock block = new DecoderBlock(manager, d, 2, 32, 1e-5f);
        NDArray prompt = manager.randomNormal(0.0f, 1.0f, new Shape(3, d), DataType.FLOAT32);
        NDArray promptMask = CausalMaskGenerator.createCausalMask(manager, 3).squeeze(0).squeeze(0);
        DecoderBlockResult prefill = block.forwardWithCache(manager, prompt, promptMask, null, null);

        NDArray next = manager.randomNormal(0.0f, 1.0f, new Shape(1, d), DataType.FLOAT32);
        NDArray wrong = manager.zeros(new Shape(1, 3), DataType.FLOAT32); // forgot to count the new token
        assertThrows(IllegalArgumentException.class,
                () -> block.forwardWithCache(manager, next, wrong, prefill.kCache, prefill.vCache));

        NDArray right = CausalMaskGenerator.createCachedMask(manager, 1, 4);
        DecoderBlockResult step = block.forwardWithCache(manager, next, right, prefill.kCache, prefill.vCache);
        assertEquals(4L, step.kCache.getShape().get(2), "cache should now hold 3 + 1 positions");
    }

    @Test
    @DisplayName("createCachedMask equals the last rows of the full causal mask")
    void testCachedMaskMatchesLastRowsOfFullMask() {
        float[] cached = CausalMaskGenerator.createCachedMask(manager, 2, 5).toFloatArray();
        float[] lastRows = CausalMaskGenerator.createCausalMask(manager, 5).squeeze(0).squeeze(0)
                .get(new ai.djl.ndarray.index.NDIndex("3:5")).toFloatArray();
        assertArrayEquals(lastRows, cached);
    }

    @Test
    @DisplayName("Passing cos/sin where the KV caches belong is rejected (the old overload silently accepted it)")
    void testCosSinAsCacheIsRejected() {
        int d = 16;
        DecoderBlock block = new DecoderBlock(manager, d, 2, 32, 1e-5f);
        NDArray x = manager.randomNormal(0.0f, 1.0f, new Shape(1, d), DataType.FLOAT32);
        NDArray notACache = manager.ones(new Shape(1, d / 2), DataType.FLOAT32);

        assertThrows(IllegalArgumentException.class,
                () -> block.forwardWithCache(manager, x, null, notACache, notACache));
        // only one of the two caches provided
        assertThrows(IllegalArgumentException.class,
                () -> block.forwardWithCache(manager, x, null, null, notACache));
    }
}
