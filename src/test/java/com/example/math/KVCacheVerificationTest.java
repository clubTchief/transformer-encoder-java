package com.example.math;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.decoder.DecoderBlock;
import com.transformer.CausalMaskGenerator;
import com.transformer.KVCacheBuffer;
import com.transformer.RopeEmbeddings;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

/**
 * KV-cache equivalence with REAL RoPE tables (not an identity rotation and not dummy ones): decoding with a cache
 * must reproduce, position by position, what one full causal pass computes. The negative control proves that the
 * test would notice if the cached path rotated by the wrong positions.
 */
public class KVCacheVerificationTest {

    private static final int D_MODEL = 16;
    private static final int HEADS = 2;
    private static final int D_K = D_MODEL / HEADS;
    private static final int D_FF = 64;
    private static final int SEQ_LEN = 6;

    private NDManager manager;
    private DecoderBlock block;
    private NDArray fullX;
    private NDArray[] rope;      // real tables for positions 0..SEQ_LEN-1, shape [SEQ_LEN, D_K]
    private float[] fullOutput;  // reference: one full causal pass, flattened [SEQ_LEN * D_MODEL]

    @BeforeEach
    void setUp() {
        manager = NDManager.newBaseManager();
        Engine.getInstance().setRandomSeed(2024);
        block = new DecoderBlock(manager, D_MODEL, HEADS, D_FF, 1e-5f);
        fullX = manager.randomNormal(0.0f, 1.0f, new Shape(SEQ_LEN, D_MODEL), DataType.FLOAT32);
        rope = RopeEmbeddings.computeCosSin(manager, SEQ_LEN, D_K);

        NDArray fullMask = CausalMaskGenerator.createCausalMask(manager, SEQ_LEN).squeeze(0).squeeze(0);
        fullOutput = block.forward(fullX, fullMask, rope[0], rope[1]).toFloatArray();
    }

    @AfterEach
    void tearDown() {
        manager.close();
    }

    private KVCacheBuffer newCache() {
        return new KVCacheBuffer(manager, 1, HEADS, SEQ_LEN, D_K);
    }

    private NDArray token(int t) {
        return fullX.get(new NDIndex(t + ":" + (t + 1)));
    }

    private float[] fullRow(int pos) {
        return Arrays.copyOfRange(fullOutput, pos * D_MODEL, (pos + 1) * D_MODEL);
    }

    private static float maxAbsDiff(float[] a, float[] b) {
        float m = 0f;
        for (int i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
        return m;
    }

    @Test
    @DisplayName("Token-by-token decoding with the correct absolute-position RoPE rows matches the full pass at EVERY position")
    public void testStepByStepMatchesFullPass() {
        KVCacheBuffer cache = newCache();
        for (int t = 0; t < SEQ_LEN; t++) {
            NDArray[] rows = RopeEmbeddings.slice(rope[0], rope[1], t, 1); // rotations for absolute position t
            NDArray mask = CausalMaskGenerator.createCachedMask(manager, 1, t + 1);
            NDArray out = block.forwardWithBuffer(manager, token(t), mask, rows[0], rows[1], cache);

            Assertions.assertEquals(t + 1, cache.getCurrentLength(), "cache length after step " + t);
            Assertions.assertArrayEquals(fullRow(t), out.toFloatArray(), 1e-4f, "position " + t);
        }
    }

    @Test
    @DisplayName("Prefill the prompt in one call, then decode: still matches the full pass at every position")
    public void testPrefillThenStepsMatchesFullPass() {
        int promptLen = 3;
        KVCacheBuffer cache = newCache();

        NDArray prompt = fullX.get(new NDIndex("0:" + promptLen));
        NDArray[] promptRows = RopeEmbeddings.slice(rope[0], rope[1], 0, promptLen);
        NDArray prefill = block.forwardWithBuffer(manager, prompt,
                CausalMaskGenerator.createCachedMask(manager, promptLen, promptLen),
                promptRows[0], promptRows[1], cache);

        float[] prefillValues = prefill.toFloatArray();
        for (int i = 0; i < promptLen; i++) {
            Assertions.assertArrayEquals(fullRow(i),
                    Arrays.copyOfRange(prefillValues, i * D_MODEL, (i + 1) * D_MODEL), 1e-4f, "prefill position " + i);
        }

        for (int t = promptLen; t < SEQ_LEN; t++) {
            NDArray[] rows = RopeEmbeddings.slice(rope[0], rope[1], t, 1);
            NDArray out = block.forwardWithBuffer(manager, token(t),
                    CausalMaskGenerator.createCachedMask(manager, 1, t + 1), rows[0], rows[1], cache);
            Assertions.assertArrayEquals(fullRow(t), out.toFloatArray(), 1e-4f, "decode position " + t);
        }
    }

    @Test
    @DisplayName("The overload that works out the RoPE rows from the cache length gives the same result")
    public void testAutomaticPositionOffset() {
        KVCacheBuffer cache = newCache();

        // a 2-token chunk first, then single tokens: the offset logic must handle both
        NDArray chunk = fullX.get(new NDIndex("0:2"));
        NDArray chunkOut = block.forwardWithBuffer(manager, chunk, CausalMaskGenerator.createCachedMask(manager, 2, 2), cache);
        float[] chunkValues = chunkOut.toFloatArray();
        for (int i = 0; i < 2; i++) {
            Assertions.assertArrayEquals(fullRow(i),
                    Arrays.copyOfRange(chunkValues, i * D_MODEL, (i + 1) * D_MODEL), 1e-4f, "chunk position " + i);
        }
        for (int t = 2; t < SEQ_LEN; t++) {
            NDArray out = block.forwardWithBuffer(manager, token(t),
                    CausalMaskGenerator.createCachedMask(manager, 1, t + 1), cache);
            Assertions.assertArrayEquals(fullRow(t), out.toFloatArray(), 1e-4f, "position " + t);
        }
    }

    @Test
    @DisplayName("Negative control: using position 0 for every step (the classic cache bug) does NOT match")
    public void testWrongPositionsAreDetected() {
        KVCacheBuffer cache = newCache();
        float[] last = null;
        for (int t = 0; t < SEQ_LEN; t++) {
            NDArray[] wrongRows = RopeEmbeddings.slice(rope[0], rope[1], 0, 1); // forgot to advance the position
            last = block.forwardWithBuffer(manager, token(t), CausalMaskGenerator.createCachedMask(manager, 1, t + 1),
                    wrongRows[0], wrongRows[1], cache).toFloatArray();
        }
        Assertions.assertTrue(maxAbsDiff(fullRow(SEQ_LEN - 1), last) > 1e-3f,
                "this test only proves something if wrong positions give a visibly different result");
    }

    @Test
    @DisplayName("Handing the WHOLE table to a single-token step is rejected instead of silently using position 0")
    public void testWholeTableInCachedStepIsRejected() {
        NDArray mask = CausalMaskGenerator.createCachedMask(manager, 1, 1);
        KVCacheBuffer cache = newCache();
        Assertions.assertThrows(IllegalArgumentException.class,
                () -> block.forwardWithBuffer(manager, token(0), mask, rope[0], rope[1], cache));
        Assertions.assertEquals(0, cache.getCurrentLength(), "a rejected call must not touch the cache");
    }
}
