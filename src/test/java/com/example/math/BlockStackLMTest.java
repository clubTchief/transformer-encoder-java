package com.example.math;

import com.transformer.BlockStackLM;
import com.transformer.CausalMaskGenerator;
import com.transformer.KVCacheBuffer;
import com.transformer.LMConfig;
import com.transformer.RopeEmbeddings;
import ai.djl.engine.Engine;
import ai.djl.training.GradientCollector;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class BlockStackLMTest {

    @Test
    public void testMultiLayerKVCacheEquivalence() {
        try (NDManager manager = NDManager.newBaseManager()) {
            int numLayers = 4;
            int dModel = 16;
            int numHeads = 2;
            int headDim = dModel / numHeads;
            int dFF = 64;
            int vocabSize = 32;
            int seqLen = 6;
            int promptLen = 3;
            float eps = 1e-5f;

            BlockStackLM model = new BlockStackLM(manager, new LMConfig(numLayers, dModel, numHeads, dFF, vocabSize, eps));

            NDArray fullX = manager.randomNormal(0.0f, 1.0f, new Shape(seqLen, dModel), DataType.FLOAT32);
            NDArray cos = manager.randomNormal(0.0f, 1.0f, new Shape(seqLen, headDim), DataType.FLOAT32);
            NDArray sin = manager.randomNormal(0.0f, 1.0f, new Shape(seqLen, headDim), DataType.FLOAT32);

            // 1. Full Sequence Forward Pass O(N^2)
            NDArray fullMask4D = CausalMaskGenerator.createCausalMask(manager, seqLen);
            NDArray fullMask = fullMask4D.squeeze(0).squeeze(0);
            NDArray fullLogits = model.forward(manager, fullX, fullMask, cos, sin);
            float[] expectedAll = fullLogits.toFloatArray();

            // 2. Incremental Execution with Layer Cache Buffers
            List<KVCacheBuffer> buffers = model.createLayerBuffers(manager, 1, seqLen);
            List<float[]> collectedLogits = new ArrayList<>();

            // A. Prefill Phase (Prompt tokens 0..3)
            NDArray promptX = fullX.get("0:" + promptLen);
            NDArray promptCos = cos.get("0:" + promptLen);
            NDArray promptSin = sin.get("0:" + promptLen);
            NDArray promptMask = CausalMaskGenerator.createCachedMask(manager, promptLen, promptLen);

            NDArray prefillLogits = model.forwardWithBuffer(manager, promptX, promptMask, promptCos, promptSin, buffers);
            float[] prefillArray = prefillLogits.toFloatArray();
            for (int i = 0; i < promptLen; i++) {
                collectedLogits.add(Arrays.copyOfRange(prefillArray, i * vocabSize, (i + 1) * vocabSize));
            }

            // B. Incremental Step Decoding Phase (Tokens 3..6)
            for (int t = promptLen; t < seqLen; t++) {
                NDArray stepX = fullX.get(t + ":" + (t + 1));
                NDArray stepCos = cos.get(t + ":" + (t + 1));
                NDArray stepSin = sin.get(t + ":" + (t + 1));
                NDArray stepMask = CausalMaskGenerator.createCachedMask(manager, 1, t + 1);

                NDArray stepLogits = model.forwardWithBuffer(manager, stepX, stepMask, stepCos, stepSin, buffers);
                collectedLogits.add(stepLogits.toFloatArray());
            }

            // 3. Verify Logits Parity at Every Position
            for (int pos = 0; pos < seqLen; pos++) {
                float[] expectedRow = Arrays.copyOfRange(expectedAll, pos * vocabSize, (pos + 1) * vocabSize);
                Assertions.assertArrayEquals(expectedRow, collectedLogits.get(pos), 1e-4f,
                        "Multi-layer logits mismatch at token position " + pos);
            }

            System.out.println("SUCCESS: Multi-layer BlockStackLM matches full pass within 1e-4 tolerance.");
        }
    }

    private static BlockStackLM smallModel(NDManager manager) {
        Engine.getInstance().setRandomSeed(42);
        return new BlockStackLM(manager, new LMConfig(2, 16, 2, 32, 13, 1e-5f));
    }

    private static NDArray causal2D(NDManager manager, int n) {
        return CausalMaskGenerator.createCausalMask(manager, n).squeeze(0).squeeze(0);
    }

    @Test
    public void testEmbeddingLivesInsideTheModelAndIsInParams() {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = smallModel(manager);
            List<NDArray> params = model.params();
            Assertions.assertSame(model.getEmbedding(), params.get(params.size() - 1), "embedding table must be the last param");
            Assertions.assertEquals(new Shape(13, 16), model.getEmbedding().getShape());
            Assertions.assertTrue(model.getEmbedding().hasGradient());
        }
    }

    @Test
    public void testForwardIdsEqualsEmbedThenForward() {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = smallModel(manager);
            NDArray ids = manager.create(new int[]{3, 1, 4, 1, 5});
            NDArray mask = causal2D(manager, 5);

            float[] viaIds = model.forwardIds(manager, ids, mask).toFloatArray();
            float[] viaHidden = model.forward(manager, model.embed(ids), mask).toFloatArray();
            Assertions.assertArrayEquals(viaIds, viaHidden, 1e-6f);
            Assertions.assertEquals(5 * 13, viaIds.length);
        }
    }

    @Test
    public void testEmbedAcceptsBatchedFloatIds() {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = smallModel(manager);
            // the dataset stores token ids as float32 [batch, seqLen]
            NDArray ids = manager.create(new float[][]{{1, 2, 3}, {4, 5, 6}});
            Assertions.assertEquals(new Shape(2, 3, 16), model.embed(ids).getShape());
        }
    }

    @Test
    public void testGradientsReachEmbeddingTableAndAttentionWeights() {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = smallModel(manager);
            NDArray mask = causal2D(manager, 4);

            try (NDManager scope = manager.newSubManager();
                 GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                NDArray logits = model.forwardIds(scope, scope.create(new int[]{1, 2, 3, 4}), mask);
                gc.backward(logits.square().sum());
            }

            List<NDArray> params = model.params();
            for (int i = 0; i < params.size(); i++) {
                float abs = params.get(i).getGradient().abs().sum().getFloat();
                Assertions.assertTrue(Float.isFinite(abs), "gradient of param #" + i + " is not finite");
            }
            // the embedding table (used for the look-up AND as the output head), and the Q/K projections of layer 0
            Assertions.assertTrue(model.getEmbedding().getGradient().abs().sum().getFloat() > 0f, "embedding table got no gradient");
            Assertions.assertTrue(params.get(0).getGradient().abs().sum().getFloat() > 0f, "wQ got no gradient");
            Assertions.assertTrue(params.get(2).getGradient().abs().sum().getFloat() > 0f, "wK got no gradient");
        }
    }

    @Test
    public void testModelIsOrderSensitiveBecauseOfRope() {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = smallModel(manager);
            NDArray mask = causal2D(manager, 5);

            // the same tokens with the first two swapped: with RoPE the last position must see a different history
            float[] a = model.forwardIds(manager, manager.create(new int[]{1, 2, 3, 4, 5}), mask).toFloatArray();
            float[] b = model.forwardIds(manager, manager.create(new int[]{2, 1, 3, 4, 5}), mask).toFloatArray();
            int last = 4 * 13;
            float maxDiff = 0f;
            for (int i = 0; i < 13; i++) maxDiff = Math.max(maxDiff, Math.abs(a[last + i] - b[last + i]));
            Assertions.assertTrue(maxDiff > 1e-5f, "last-position logits ignore the order of earlier tokens: RoPE is not active");
        }
    }

    @Test
    public void testAutomaticRopeOffsetMatchesFullPassAtEveryPosition() {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = smallModel(manager);
            int seqLen = 7, promptLen = 3, vocab = 13;
            NDArray hidden = manager.randomNormal(0.0f, 1.0f, new Shape(seqLen, 16), DataType.FLOAT32);

            float[] full = model.forward(manager, hidden, causal2D(manager, seqLen)).toFloatArray();

            List<KVCacheBuffer> buffers = model.createLayerBuffers(manager, 1, seqLen);
            List<float[]> rows = new ArrayList<>();
            float[] prefill = model.forwardWithBuffer(manager, hidden.get("0:" + promptLen),
                    CausalMaskGenerator.createCachedMask(manager, promptLen, promptLen), buffers).toFloatArray();
            for (int i = 0; i < promptLen; i++) rows.add(Arrays.copyOfRange(prefill, i * vocab, (i + 1) * vocab));
            for (int t = promptLen; t < seqLen; t++) {
                rows.add(model.forwardWithBuffer(manager, hidden.get(t + ":" + (t + 1)),
                        CausalMaskGenerator.createCachedMask(manager, 1, t + 1), buffers).toFloatArray());
            }
            for (int pos = 0; pos < seqLen; pos++) {
                Assertions.assertArrayEquals(Arrays.copyOfRange(full, pos * vocab, (pos + 1) * vocab), rows.get(pos), 1e-4f,
                        "position " + pos);
            }
        }
    }

    @Test
    public void testOutOfSyncLayerCachesAreRejected() {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = smallModel(manager);
            List<KVCacheBuffer> buffers = model.createLayerBuffers(manager, 1, 8);
            buffers.get(0).update(manager.zeros(new Shape(1, 2, 1, 8)), manager.zeros(new Shape(1, 2, 1, 8)));
            NDArray x = manager.randomNormal(0.0f, 1.0f, new Shape(1, 16), DataType.FLOAT32);
            Assertions.assertThrows(IllegalStateException.class,
                    () -> model.forwardWithBuffer(manager, x, null, buffers));
        }
    }

    @Test
    public void testParamNamesMatchParams() {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = smallModel(manager);
            List<String> names = model.paramNames();
            Assertions.assertEquals(model.params().size(), names.size());
            Assertions.assertEquals("layer0.wQ", names.get(0));
            Assertions.assertEquals("finalNorm.gamma", names.get(names.size() - 2));
            Assertions.assertEquals("embedding", names.get(names.size() - 1));
        }
    }

    private static double std(NDArray array) {
        float[] v = array.toFloatArray();
        double mean = 0;
        for (float x : v) mean += x;
        mean /= v.length;
        double var = 0;
        for (float x : v) var += (x - mean) * (x - mean);
        return Math.sqrt(var / v.length);
    }

    @Test
    public void testGptStyleInitScalesResidualWritersWithDepth() {
        try (NDManager manager = NDManager.newBaseManager()) {
            Engine.getInstance().setRandomSeed(8);
            BlockStackLM deep = new BlockStackLM(manager, new LMConfig(8, 64, 4, 128, 20, 1e-5f), 0.02f);
            List<String> names = deep.paramNames();
            List<NDArray> params = deep.params();

            Assertions.assertEquals(0.02, std(params.get(names.indexOf("layer0.wQ"))), 0.002);
            // 0.02 / sqrt(2 * 8 layers) = 0.005
            Assertions.assertEquals(0.005, std(params.get(names.indexOf("layer0.wO"))), 0.0006);
            Assertions.assertEquals(0.005, std(params.get(names.indexOf("layer7.wD"))), 0.0006);
        }
    }

    @Test
    public void testGptStyleModelProducesFiniteLogits() {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = new BlockStackLM(manager, new LMConfig(4, 32, 2, 64, 15, 1e-5f), 0.02f);
            float[] logits = model.forwardIds(manager, manager.create(new int[]{1, 2, 3, 4, 5, 6}), causal2D(manager, 6)).toFloatArray();
            for (float v : logits) Assertions.assertTrue(Float.isFinite(v));
        }
    }
}
