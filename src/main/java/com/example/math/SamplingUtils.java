package com.example.math;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;

import java.util.Random;

public class SamplingUtils {

    private static final Random random = new Random();

    /**
     * Takes logits for the last token [vocabSize], applies temperature scaling,
     * performs Top-K filtering, and samples a single token ID.
     */
    public static int sampleNextToken(NDArray logits, float temperature, int topK) {
        // 1. Temperature Scaling
        NDArray scaledLogits = logits.div(Math.max(temperature, 1e-5f));

        // 2. Top-K Filtering
        float[] logitsArray = scaledLogits.toFloatArray();
        int vocabSize = logitsArray.length;
        topK = Math.min(topK, vocabSize);

        // Find indices of top K values
        Integer[] indices = new Integer[vocabSize];
        for (int i = 0; i < vocabSize; i++) indices[i] = i;

        java.util.Arrays.sort(indices, (a, b) -> Float.compare(logitsArray[b], logitsArray[a]));

        // Mask tokens outside Top-K with -infinity
        float[] filteredLogits = new float[vocabSize];
        java.util.Arrays.fill(filteredLogits, Float.NEGATIVE_INFINITY);
        for (int i = 0; i < topK; i++) {
            int idx = indices[i];
            filteredLogits[idx] = logitsArray[idx];
        }

        // 3. Compute Softmax Probabilities over Top-K
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < topK; i++) {
            if (filteredLogits[indices[i]] > maxLogit) {
                maxLogit = filteredLogits[indices[i]];
            }
        }

        float sumExp = 0.0f;
        float[] probs = new float[vocabSize];
        for (int i = 0; i < topK; i++) {
            int idx = indices[i];
            probs[idx] = (float) Math.exp(filteredLogits[idx] - maxLogit);
            sumExp += probs[idx];
        }

        for (int i = 0; i < topK; i++) {
            int idx = indices[i];
            probs[idx] /= sumExp;
        }

        // 4. Random Categorical Sampling
        float r = random.nextFloat();
        float cumulativeProbability = 0.0f;
        for (int i = 0; i < topK; i++) {
            int idx = indices[i];
            cumulativeProbability += probs[idx];
            if (r <= cumulativeProbability) {
                return idx;
            }
        }

        return indices[0]; // Fallback to argmax
    }
}