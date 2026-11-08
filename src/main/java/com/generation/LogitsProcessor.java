package com.generation;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import java.util.Arrays;
import java.util.List;

public class LogitsProcessor {

    /**
     * Applies temperature scaling to raw logits.
     */
    public static NDArray applyTemperature(NDArray logits, float temperature) {
        if (temperature <= 0.0f) {
            throw new IllegalArgumentException("Temperature must be greater than 0");
        }
        if (temperature == 1.0f) {
            return logits;
        }
        return logits.div(temperature);
    }

    /**
     * Applies repetition penalty to logits based on previously generated token IDs.
     */
    public static NDArray applyRepetitionPenalty(NDArray logits, List<Integer> generatedTokens, float penalty) {
        if (penalty == 1.0f || generatedTokens.isEmpty()) {
            return logits;
        }

        NDArray penalizedLogits = logits.duplicate();
        float[] logitArray = penalizedLogits.toFloatArray();

        for (int tokenId : generatedTokens) {
            if (tokenId >= 0 && tokenId < logitArray.length) {
                float val = logitArray[tokenId];
                logitArray[tokenId] = val < 0 ? val * penalty : val / penalty;
            }
        }

        penalizedLogits.set(logitArray);
        return penalizedLogits;
    }

    /**
     * Top-K filtering: keeps only top K highest logits, setting the rest to -infinity.
     */
    public static NDArray applyTopK(NDArray logits, int k) {
        long vocabSize = logits.getShape().get(logits.getShape().dimension() - 1);
        if (k <= 0 || k >= vocabSize) {
            return logits;
        }

        // Sort ascending, pick top-K threshold from end of sorted array
        NDArray sorted = logits.sort(-1);
        float[] sortedVals = sorted.toFloatArray();
        float threshold = sortedVals[sortedVals.length - k];

        float[] originalVals = logits.toFloatArray();
        float[] filteredVals = new float[originalVals.length];

        for (int i = 0; i < originalVals.length; i++) {
            filteredVals[i] = (originalVals[i] >= threshold) ? originalVals[i] : Float.NEGATIVE_INFINITY;
        }

        NDArray result = logits.getManager().create(filteredVals, logits.getShape());
        return result;
    }

    /**
     * Top-P (Nucleus) filtering: keeps cumulative probability mass up to p.
     */
    public static NDArray applyTopP(NDArray logits, float p) {
        if (p >= 1.0f || p <= 0.0f) {
            return logits;
        }

        float[] logitVals = logits.toFloatArray();
        int vocabSize = logitVals.length;

        // Pair token indices with logit values
        Integer[] indices = new Integer[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            indices[i] = i;
        }

        // Sort indices by logit value descending
        Arrays.sort(indices, (a, b) -> Float.compare(logitVals[b], logitVals[a]));

        // Calculate Softmax probabilities over sorted indices
        double maxLogit = logitVals[indices[0]];
        double sumExp = 0.0;
        double[] probs = new double[vocabSize];

        for (int i = 0; i < vocabSize; i++) {
            probs[i] = Math.exp(logitVals[indices[i]] - maxLogit);
            sumExp += probs[i];
        }

        // Apply Top-P cumulative threshold
        float[] filteredLogits = new float[vocabSize];
        Arrays.fill(filteredLogits, Float.NEGATIVE_INFINITY);

        double cumProb = 0.0;
        for (int i = 0; i < vocabSize; i++) {
            int tokenIdx = indices[i];
            filteredLogits[tokenIdx] = logitVals[tokenIdx];
            cumProb += (probs[i] / sumExp);
            if (cumProb >= p) {
                break;
            }
        }

        return logits.getManager().create(filteredLogits, logits.getShape());
    }
}