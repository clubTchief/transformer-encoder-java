package com.example.math;

import com.generation.LogitsProcessor;
import com.generation.Sampler;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import java.util.List;

public class SamplerTest {

    @Test
    public void testLogitsProcessingAndSampling() {
        try (NDManager manager = NDManager.newBaseManager()) {
            // Vocab size = 5
            NDArray rawLogits = manager.create(new float[]{1.0f, 2.0f, 5.0f, 0.5f, 0.1f});

            // 1. Greedy sampling check
            Sampler sampler = new Sampler(42);
            int greedyToken = sampler.sampleGreedy(rawLogits);
            Assertions.assertEquals(2, greedyToken, "Greedy sampling must select highest logit index (2)");

            // 2. Temperature scaling
            NDArray scaledLogits = LogitsProcessor.applyTemperature(rawLogits, 0.5f);
            Assertions.assertEquals(10.0f, scaledLogits.getFloat(2), 1e-4f, "Logit at idx 2 should be doubled (5.0 / 0.5)");

            // 3. Repetition penalty
            NDArray penalizedLogits = LogitsProcessor.applyRepetitionPenalty(rawLogits, List.of(2), 2.0f);
            Assertions.assertEquals(2.5f, penalizedLogits.getFloat(2), 1e-4f, "Logit at idx 2 should be penalized (5.0 / 2.0)");

            // 4. Top-K filtering (k = 2)
            NDArray topKLogits = LogitsProcessor.applyTopK(rawLogits, 2);
            float[] topKArray = topKLogits.toFloatArray();
            Assertions.assertTrue(Float.isInfinite(topKArray[0]) && topKArray[0] < 0, "Idx 0 should be -inf");
            Assertions.assertEquals(2.0f, topKArray[1], 1e-4f);
            Assertions.assertEquals(5.0f, topKArray[2], 1e-4f);

            // 5. Multinomial sampling execution
            int sampleToken = sampler.sampleMultinomial(topKLogits);
            Assertions.assertTrue(sampleToken == 1 || sampleToken == 2, "Sampled token must be within Top-K set");

            System.out.println("SUCCESS: Logits processing and sampling verified.");
        }
    }

    @Test
    public void testNeverSamplesFilteredTokens() {
        try (NDManager manager = NDManager.newBaseManager()) {
            float ninf = Float.NEGATIVE_INFINITY;
            NDArray logits = manager.create(new float[]{ninf, 0.0f, ninf, 1.0f, ninf});
            Sampler sampler = new Sampler(7);
            for (int i = 0; i < 5000; i++) {
                int token = sampler.sampleMultinomial(logits);
                Assertions.assertTrue(token == 1 || token == 3, "sampled a filtered-out token: " + token);
            }
        }
    }
}
