package com.example.math;

import com.decoder.DecoderBlock;
import com.decoder.DecoderBlockResult;
import com.transformer.CausalMaskGenerator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class KVCacheVerificationTest {

    @Test
    public void testKVCacheEquivalence() {
        try (NDManager manager = NDManager.newBaseManager()) {
            int dModel = 16;
            int numHeads = 2;
            int dK = dModel / numHeads; // dK = 8
            int dFF = 64;
            float eps = 1e-5f;
            int seqLen = 5;

            System.out.println("================================================================================");
            System.out.println("            >>> RUNNING KV-CACHE EQUIVALENCE VERIFICATION <<<                   ");
            System.out.println("================================================================================");
            System.out.printf("dModel: %d | numHeads: %d | dK: %d | Sequence Length: %d\n", dModel, numHeads, dK, seqLen);
            System.out.println("--------------------------------------------------------------------------------");

            DecoderBlock block = new DecoderBlock(manager, dModel, numHeads, dFF, eps);

            // Inputs and pre-computed RoPE cos/sin tensors matching head dimension dK
            NDArray fullX = manager.randomNormal(0.0f, 1.0f, new Shape(seqLen, dModel), DataType.FLOAT32);
            NDArray cos = manager.ones(new Shape(seqLen, dK), DataType.FLOAT32);
            NDArray sin = manager.zeros(new Shape(seqLen, dK), DataType.FLOAT32);

            // 1. Compute Full Non-Cached Forward Pass O(N^2)
            NDArray fullMask4D = CausalMaskGenerator.createCausalMask(manager, seqLen);
            NDArray fullMask = fullMask4D.squeeze(0).squeeze(0);
            NDArray fullOutput = block.forward(fullX, fullMask, cos, sin);

            System.out.println("Step 1: Full sequence pass executed successfully.");
            System.out.println("        Full output shape: " + fullOutput.getShape());

            // 2. Incremental KV-Cached Step Generation O(1)
            NDArray kCache = null;
            NDArray vCache = null;
            NDArray cachedLastOutput = null;
            java.util.List<float[]> stepOutputs = new java.util.ArrayList<>();

            System.out.println("\nStep 2: Simulating incremental step-by-step KV-Cached inference...");

            for (int t = 0; t < seqLen; t++) {
                NDArray singleTokenX = fullX.get(new ai.djl.ndarray.index.NDIndex("{}:{}", t, t + 1));
                NDArray stepCos = cos.get(new ai.djl.ndarray.index.NDIndex("{}:{}", t, t + 1));
                NDArray stepSin = sin.get(new ai.djl.ndarray.index.NDIndex("{}:{}", t, t + 1));

                NDArray stepMask = CausalMaskGenerator.createCachedMask(manager, 1, t + 1);

                DecoderBlockResult res = block.forwardWithCache(
                        singleTokenX, stepMask, stepCos, stepSin, kCache, vCache
                );

                kCache = res.kCache;
                vCache = res.vCache;
                cachedLastOutput = res.output;
                stepOutputs.add(res.output.toFloatArray());

                System.out.printf("  [Step %d/%d] Token processed | K-Cache shape: %s | V-Cache shape: %s\n",
                        t + 1, seqLen, kCache.getShape(), vCache.getShape());
            }

            // 3. Compare outputs at position (seqLen - 1)
            NDArray fullLastOutput = fullOutput.get(new ai.djl.ndarray.index.NDIndex("{}:{}", seqLen - 1, seqLen));

            float[] expected = fullLastOutput.toFloatArray();
            float[] actual = cachedLastOutput.toFloatArray();

            System.out.println("--------------------------------------------------------------------------------");
            System.out.println("Comparing full sequence vector vs KV-cached final vector:");
            System.out.printf("  Full non-cached sample:   [%.5f, %.5f, %.5f, ...]\n", expected[0], expected[1], expected[2]);
            System.out.printf("  KV-cached step sample:    [%.5f, %.5f, %.5f, ...]\n", actual[0], actual[1], actual[2]);

            Assertions.assertArrayEquals(expected, actual, 1e-4f, "KV-Cache logits differ from full calculation!");

            // Every position must match, not just the last one
            float[] fullAll = fullOutput.toFloatArray();
            for (int pos = 0; pos < seqLen; pos++) {
                float[] row = java.util.Arrays.copyOfRange(fullAll, pos * dModel, (pos + 1) * dModel);
                Assertions.assertArrayEquals(row, stepOutputs.get(pos), 1e-4f, "Cached step " + pos + " differs from full pass");
            }

            System.out.println("--------------------------------------------------------------------------------");
            System.out.println("SUCCESS: KV-Cache output matches full O(N^2) pass within 1e-4 tolerance.");
            System.out.println("================================================================================\n");
        }
    }

    @Test
    public void testPrefillThenStepsMatchesFullPass() {
        try (NDManager manager = NDManager.newBaseManager()) {
            int dModel = 16, numHeads = 2, dFF = 64, seqLen = 6, promptLen = 3;
            DecoderBlock block = new DecoderBlock(manager, dModel, numHeads, dFF, 1e-5f);
            NDArray fullX = manager.randomNormal(0.0f, 1.0f, new Shape(seqLen, dModel), DataType.FLOAT32);
            NDArray dummy = manager.ones(new Shape(seqLen, dModel / numHeads), DataType.FLOAT32); // cos/sin are unused
            NDArray fullMask = CausalMaskGenerator.createCausalMask(manager, seqLen).squeeze(0).squeeze(0);
            float[] full = block.forward(fullX, fullMask, dummy, dummy).toFloatArray();

            // 1. Prefill: the whole prompt in one call, building the cache
            NDArray prompt = fullX.get(new ai.djl.ndarray.index.NDIndex("0:" + promptLen));
            DecoderBlockResult res = block.forwardWithCache(prompt,
                    CausalMaskGenerator.createCachedMask(manager, promptLen, promptLen), dummy, dummy, null, null);
            java.util.List<float[]> rows = new java.util.ArrayList<>();
            float[] prefill = res.output.toFloatArray();
            for (int i = 0; i < promptLen; i++) rows.add(java.util.Arrays.copyOfRange(prefill, i * dModel, (i + 1) * dModel));

            // 2. Decode the remaining tokens one at a time
            for (int t = promptLen; t < seqLen; t++) {
                NDArray x = fullX.get(new ai.djl.ndarray.index.NDIndex(t + ":" + (t + 1)));
                res = block.forwardWithCache(x, CausalMaskGenerator.createCachedMask(manager, 1, t + 1),
                        dummy, dummy, res.kCache, res.vCache);
                rows.add(res.output.toFloatArray());
            }

            for (int pos = 0; pos < seqLen; pos++) {
                float[] expected = java.util.Arrays.copyOfRange(full, pos * dModel, (pos + 1) * dModel);
                Assertions.assertArrayEquals(expected, rows.get(pos), 1e-4f, "position " + pos);
            }
        }
    }
}
