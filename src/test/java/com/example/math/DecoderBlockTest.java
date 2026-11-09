package com.example.math;

import com.decoder.DecoderBlock;
import com.transformer.GradientEngine;
import com.transformer.CausalMaskGenerator;
import com.transformer.KVCacheBuffer;
import com.transformer.RopeEmbeddings;
import ai.djl.engine.Engine;
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
        NDArray[] ropeTables = RopeEmbeddings.computeCosSin(manager, seqLen, dK);
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
        NDArray[] rope = RopeEmbeddings.computeCosSin(manager, n, d / 2);

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
        NDArray[] rope = RopeEmbeddings.computeCosSin(manager, n, d / 2);
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
    @DisplayName("createCachedMask equals the last rows of the full causal mask")
    void testCachedMaskMatchesLastRowsOfFullMask() {
        float[] cached = CausalMaskGenerator.createCachedMask(manager, 2, 5).toFloatArray();
        float[] lastRows = CausalMaskGenerator.createCausalMask(manager, 5).squeeze(0).squeeze(0)
                .get(new ai.djl.ndarray.index.NDIndex("3:5")).toFloatArray();
        assertArrayEquals(lastRows, cached);
    }

    @Test
    @DisplayName("A cached step needs a mask of exactly [newTokens, cached + new]; a rejected call leaves the cache untouched")
    void testCachedStepRequiresExactMask() {
        int d = 16, heads = 2;
        DecoderBlock block = new DecoderBlock(manager, d, heads, 32, 1e-5f);
        KVCacheBuffer cache = new KVCacheBuffer(manager, 1, heads, 8, d / heads);
        NDArray prompt = manager.randomNormal(0.0f, 1.0f, new Shape(3, d), DataType.FLOAT32);
        block.forwardWithBuffer(manager, prompt, CausalMaskGenerator.createCachedMask(manager, 3, 3), cache);
        assertEquals(3, cache.getCurrentLength());

        NDArray next = manager.randomNormal(0.0f, 1.0f, new Shape(1, d), DataType.FLOAT32);
        NDArray wrong = manager.zeros(new Shape(1, 3), DataType.FLOAT32); // forgot to count the new token
        assertThrows(IllegalArgumentException.class, () -> block.forwardWithBuffer(manager, next, wrong, cache));
        assertEquals(3, cache.getCurrentLength(), "a rejected call must not have written to the cache");

        block.forwardWithBuffer(manager, next, CausalMaskGenerator.createCachedMask(manager, 1, 4), cache);
        assertEquals(4, cache.getCurrentLength());
    }

    @Test
    @DisplayName("Overflowing the KV cache is rejected and leaves the cache untouched")
    void testCacheOverflowRejected() {
        int d = 16, heads = 2;
        DecoderBlock block = new DecoderBlock(manager, d, heads, 32, 1e-5f);
        KVCacheBuffer cache = new KVCacheBuffer(manager, 1, heads, 4, d / heads); // room for 4 positions
        NDArray three = manager.randomNormal(0.0f, 1.0f, new Shape(3, d), DataType.FLOAT32);
        block.forwardWithBuffer(manager, three, CausalMaskGenerator.createCachedMask(manager, 3, 3), cache);

        NDArray two = manager.randomNormal(0.0f, 1.0f, new Shape(2, d), DataType.FLOAT32);
        NDArray mask = CausalMaskGenerator.createCachedMask(manager, 2, 5);
        assertThrows(IllegalStateException.class, () -> block.forwardWithBuffer(manager, two, mask, cache));
        assertEquals(3, cache.getCurrentLength());
    }

    @Test
    @DisplayName("RoPE tables are mandatory and must cover exactly the tokens being processed")
    void testRopeTablesAreMandatoryAndExact() {
        int d = 16, heads = 2, hd = d / heads, n = 4;
        DecoderBlock block = new DecoderBlock(manager, d, heads, 32, 1e-5f);
        NDArray x = manager.randomNormal(0.0f, 1.0f, new Shape(n, d), DataType.FLOAT32);
        NDArray mask = CausalMaskGenerator.createCausalMask(manager, n).squeeze(0).squeeze(0);
        NDArray[] whole = RopeEmbeddings.computeCosSin(manager, 8, hd); // 8 positions, but only 4 tokens

        assertThrows(IllegalArgumentException.class, () -> block.forward(x, mask, null, null)); // there is no "no RoPE" mode
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> block.forward(x, mask, whole[0], whole[1]));
        assertTrue(e.getMessage().contains("RopeEmbeddings.slice"), "message should point to slice(): " + e.getMessage());

        // the classic trap: handing the whole table to a single-token step would silently use position 0
        NDArray one = manager.randomNormal(0.0f, 1.0f, new Shape(1, d), DataType.FLOAT32);
        assertThrows(IllegalArgumentException.class, () -> block.forward(one, null, whole[0], whole[1]));
    }

    @Test
    @DisplayName("The convenience overload that builds its own tables equals the explicit one")
    void testOwnTablesEqualExplicitTables() {
        int d = 16, heads = 2, n = 5;
        DecoderBlock block = new DecoderBlock(manager, d, heads, 32, 1e-5f);
        NDArray x = manager.randomNormal(0.0f, 1.0f, new Shape(n, d), DataType.FLOAT32);
        NDArray mask = CausalMaskGenerator.createCausalMask(manager, n).squeeze(0).squeeze(0);
        NDArray[] t = RopeEmbeddings.computeCosSin(manager, n, d / heads);

        assertArrayEquals(block.forward(x, mask, t[0], t[1]).toFloatArray(),
                block.forward(manager, x, mask).toFloatArray(), 1e-6f);
    }

    private static float[] lastRow(NDArray out, int n, int d) {
        float[] all = out.toFloatArray();
        return java.util.Arrays.copyOfRange(all, (n - 1) * d, n * d);
    }

    private static float maxAbsDiff(float[] a, float[] b) {
        float m = 0f;
        for (int i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
        return m;
    }

    @Test
    @DisplayName("RoPE gives the block a sense of order: swapping two EARLIER tokens changes the last position")
    void testRopeMakesTheBlockOrderSensitive() {
        Engine.getInstance().setRandomSeed(11);
        int n = 5, d = 16, heads = 2, hd = d / heads;
        DecoderBlock block = new DecoderBlock(manager, d, heads, 32, 1e-5f);
        NDArray mask = CausalMaskGenerator.createCausalMask(manager, n).squeeze(0).squeeze(0);

        float[][] a = new float[n][d];
        for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) a[i][j] = (float) Math.sin(i * 1.7 + j * 0.9);
        float[][] b = new float[n][];
        for (int i = 0; i < n; i++) b[i] = a[i].clone();
        b[0] = a[1].clone();
        b[1] = a[0].clone(); // the same tokens, with tokens 0 and 1 swapped

        NDArray[] real = RopeEmbeddings.computeCosSin(manager, n, hd);
        float[] withA = lastRow(block.forward(manager.create(a), mask, real[0], real[1]), n, d);
        float[] withB = lastRow(block.forward(manager.create(b), mask, real[0], real[1]), n, d);
        assertTrue(maxAbsDiff(withA, withB) > 1e-3f, "with RoPE the order of earlier tokens must matter");

        // Control: identity tables (cos = 1, sin = 0) rotate nothing, so the last token sees an unordered set
        NDArray identityCos = manager.ones(new Shape(n, hd), DataType.FLOAT32);
        NDArray identitySin = manager.zeros(new Shape(n, hd), DataType.FLOAT32);
        float[] idA = lastRow(block.forward(manager.create(a), mask, identityCos, identitySin), n, d);
        float[] idB = lastRow(block.forward(manager.create(b), mask, identityCos, identitySin), n, d);
        assertTrue(maxAbsDiff(idA, idB) < 1e-4f, "without RoPE the order of earlier tokens cannot matter here");
    }
}
