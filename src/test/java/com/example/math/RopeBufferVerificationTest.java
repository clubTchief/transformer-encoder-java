package com.example.math;

import com.decoder.DecoderBlock;
import com.transformer.CausalMaskGenerator;
import com.transformer.KVCacheBuffer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class RopeBufferVerificationTest {

    @Test
    public void testRopeAndKVCacheBufferEquivalence() {
        try (NDManager manager = NDManager.newBaseManager()) {
            int dModel = 16;
            int numHeads = 2;
            int headDim = dModel / numHeads;
            int dFF = 64;
            int seqLen = 5;
            float eps = 1e-5f;

            DecoderBlock block = new DecoderBlock(manager, dModel, numHeads, dFF, eps);

            NDArray fullX = manager.randomNormal(0.0f, 1.0f, new Shape(seqLen, dModel), DataType.FLOAT32);

            // Generate realistic RoPE frequencies
            NDArray cos = manager.randomNormal(0.0f, 1.0f, new Shape(seqLen, headDim), DataType.FLOAT32);
            NDArray sin = manager.randomNormal(0.0f, 1.0f, new Shape(seqLen, headDim), DataType.FLOAT32);

            // 1. Full Sequence Forward Pass with RoPE
            NDArray fullMask4D = CausalMaskGenerator.createCausalMask(manager, seqLen);
            NDArray fullMask = fullMask4D.squeeze(0).squeeze(0);
            NDArray fullOutput = block.forward(manager, fullX, fullMask, cos, sin);

            // 2. Step-by-Step Incremental Pass using In-Place KVCacheBuffer
            KVCacheBuffer cacheBuffer = new KVCacheBuffer(manager, 1, numHeads, seqLen, headDim);
            java.util.List<float[]> stepOutputs = new java.util.ArrayList<>();

            for (int t = 0; t < seqLen; t++) {
                NDArray singleX = fullX.get(new ai.djl.ndarray.index.NDIndex("{}:{}", t, t + 1));
                NDArray stepCos = cos.get(new ai.djl.ndarray.index.NDIndex("{}:{}", t, t + 1));
                NDArray stepSin = sin.get(new ai.djl.ndarray.index.NDIndex("{}:{}", t, t + 1));
                NDArray stepMask = CausalMaskGenerator.createCachedMask(manager, 1, t + 1);

                NDArray stepOut = block.forwardWithBuffer(manager, singleX, stepMask, stepCos, stepSin, cacheBuffer);
                stepOutputs.add(stepOut.toFloatArray());
            }

            // 3. Verify Numerical Equivalence across all token positions
            float[] fullAll = fullOutput.toFloatArray();
            for (int pos = 0; pos < seqLen; pos++) {
                float[] expectedRow = java.util.Arrays.copyOfRange(fullAll, pos * dModel, (pos + 1) * dModel);
                Assertions.assertArrayEquals(expectedRow, stepOutputs.get(pos), 1e-4f, 
                        "Mismatch found at position " + pos);
            }

            System.out.println("SUCCESS: In-place KV-Buffer with RoPE matches full forward pass within 1e-4 tolerance.");
        }
    }
}