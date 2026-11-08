package com.example.math;

import com.transformer.BlockStackLM;
import com.transformer.CausalMaskGenerator;
import com.transformer.KVCacheBuffer;
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

            BlockStackLM model = new BlockStackLM(manager, numLayers, dModel, numHeads, dFF, vocabSize, eps);

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
}