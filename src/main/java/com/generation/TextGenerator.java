package com.generation;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.transformer.BlockStackLM;
import com.transformer.CausalMaskGenerator;
import com.transformer.KVCacheBuffer;
import com.transformer.RopeEmbeddings;

import java.util.ArrayList;
import java.util.List;

/**
 * Autoregressive generation with per-layer KV caches: the prompt is processed once (prefill), then every
 * sampled token is embedded, fed back in, and attends to the cached keys/values of everything before it.
 */
public class TextGenerator {

    private final BlockStackLM model;
    private final Sampler sampler;

    public TextGenerator(BlockStackLM model, Sampler sampler) {
        this.model = model;
        this.sampler = sampler;
    }

    /**
     * @param manager           owns the KV buffers and the RoPE tables; per-step tensors use short-lived sub-managers
     * @param promptIds         token ids of the prompt (at least one)
     * @param maxGenTokens      maximum number of new tokens
     * @param eosTokenId        stop after sampling this id (it is included in the result); -1 disables
     * @param temperature       sampling temperature; values &lt;= 0 mean greedy decoding
     * @param topK              keep only the K most likely tokens (0 disables)
     * @param topP              nucleus threshold (1.0 disables)
     * @param repetitionPenalty penalty for already generated tokens (1.0 disables)
     * @return the generated token ids (without the prompt)
     */
    public List<Integer> generate(NDManager manager,
                                  int[] promptIds,
                                  int maxGenTokens,
                                  int eosTokenId,
                                  float temperature,
                                  int topK,
                                  float topP,
                                  float repetitionPenalty) {
        if (promptIds.length == 0) {
            throw new IllegalArgumentException("The prompt needs at least one token");
        }
        List<Integer> generated = new ArrayList<>();
        if (maxGenTokens < 1) {
            return generated;
        }

        int promptLen = promptIds.length;
        int maxSeqLen = promptLen + maxGenTokens;
        List<KVCacheBuffer> buffers = model.createLayerBuffers(manager, 1, maxSeqLen);
        NDArray[] rope = RopeEmbeddings.computeCosSin(manager, maxSeqLen, model.getHeadDim());

        // 1. Prefill: the whole prompt in one pass, filling the caches
        int next;
        try (NDManager step = manager.newSubManager()) {
            NDArray hidden = model.embed(step.create(promptIds));                       // [promptLen, dModel]
            NDArray[] rows = RopeEmbeddings.slice(rope[0], rope[1], 0, promptLen);
            rows[0].attach(step);
            rows[1].attach(step);
            NDArray mask = CausalMaskGenerator.createCachedMask(step, promptLen, promptLen);
            NDArray logits = model.forwardWithBuffer(step, hidden, mask, rows[0], rows[1], buffers);
            next = sampleNextToken(logits.get((long) (promptLen - 1)), generated, temperature, topK, topP, repetitionPenalty);
        }
        generated.add(next);

        // 2. Decode: feed the token that was just sampled, one position at a time
        while (next != eosTokenId && generated.size() < maxGenTokens) {
            int position = promptLen + generated.size() - 1; // absolute position of the token being fed
            try (NDManager step = manager.newSubManager()) {
                NDArray hidden = model.embed(step.create(new int[]{next}));              // [1, dModel]
                NDArray[] rows = RopeEmbeddings.slice(rope[0], rope[1], position, 1);
                rows[0].attach(step);
                rows[1].attach(step);
                NDArray mask = CausalMaskGenerator.createCachedMask(step, 1, position + 1);
                NDArray logits = model.forwardWithBuffer(step, hidden, mask, rows[0], rows[1], buffers); // [1, vocab]
                next = sampleNextToken(logits.get(0L), generated, temperature, topK, topP, repetitionPenalty);
            }
            generated.add(next);
        }
        return generated;
    }

    private int sampleNextToken(NDArray rawLogits, List<Integer> history, float temperature,
                                int topK, float topP, float repetitionPenalty) {
        NDArray processed = LogitsProcessor.applyRepetitionPenalty(rawLogits, history, repetitionPenalty);

        if (temperature <= 0.0f) {
            return sampler.sampleGreedy(processed);
        }
        processed = LogitsProcessor.applyTemperature(processed, temperature);
        if (topK > 0) {
            processed = LogitsProcessor.applyTopK(processed, topK);
        }
        if (topP < 1.0f) {
            processed = LogitsProcessor.applyTopP(processed, topP);
        }
        return sampler.sampleMultinomial(processed);
    }
}
