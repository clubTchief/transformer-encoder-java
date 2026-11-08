package com.transformer;

import java.util.*;

public class SamplingUtils {

    /**
     * Samples the next token ID using Temperature scaling and Top-p (Nucleus) / Top-k filtering.
     *
     * @param logits      Raw unnormalized logits array from the model [vocabSize]
     * @param temperature Controls randomness (T < 1.0 = focused, T > 1.0 = creative)
     * @param topK        Limits candidate pool to top K highest logits (set <= 0 to disable)
     * @param topP        Nucleus threshold: cumulative probability cutoff (e.g., 0.9)
     * @return Selected token ID
     */
    public static int sampleToken(float[] logits, float temperature, int topK, float topP) {
        int vocabSize = logits.length;
        
        // 1. Apply Temperature Scaling
        float[] scaledLogits = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            scaledLogits[i] = logits[i] / Math.max(temperature, 1e-5f);
        }

        // 2. Pair indices with logits for sorting
        List<TokenLogit> tokenLogits = new ArrayList<>(vocabSize);
        for (int i = 0; i < vocabSize; i++) {
            tokenLogits.add(new TokenLogit(i, scaledLogits[i]));
        }
        tokenLogits.sort((a, b) -> Float.compare(b.logit, a.logit));

        // 3. Apply Top-K Truncation
        if (topK > 0 && topK < tokenLogits.size()) {
            tokenLogits = tokenLogits.subList(0, topK);
        }

        // 4. Compute Softmax Probabilities over candidates
        float maxLogit = tokenLogits.get(0).logit;
        double sumExp = 0.0;
        double[] probs = new double[tokenLogits.size()];

        for (int i = 0; i < tokenLogits.size(); i++) {
            probs[i] = Math.exp(tokenLogits.get(i).logit - maxLogit); // Stable softmax
            sumExp += probs[i];
        }

        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sumExp;
        }

        // 5. Apply Top-P (Nucleus) Truncation
        double cumProb = 0.0;
        int cutoffIndex = probs.length;
        for (int i = 0; i < probs.length; i++) {
            cumProb += probs[i];
            if (cumProb >= topP) {
                cutoffIndex = i + 1;
                break;
            }
        }
        
        // Truncate to nucleus
        List<TokenLogit> nucleusTokens = tokenLogits.subList(0, cutoffIndex);
        double nucleusSum = 0.0;
        for (int i = 0; i < cutoffIndex; i++) {
            nucleusSum += probs[i];
        }

        // 6. Stochastic Categorical Sampling
        double r = Math.random() * nucleusSum;
        double accum = 0.0;
        for (int i = 0; i < cutoffIndex; i++) {
            accum += probs[i];
            if (accum >= r) {
                return nucleusTokens.get(i).id;
            }
        }

        return nucleusTokens.get(0).id;
    }

    private static class TokenLogit {
        final int id;
        final float logit;

        TokenLogit(int id, float logit) {
            this.id = id;
            this.logit = logit;
        }
    }
}