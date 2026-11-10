package com.generation;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.transformer.BlockStackLM;
import com.transformer.CausalMaskGenerator;
import com.transformer.KVCacheBuffer;
import com.transformer.RopeEmbeddings;

import java.util.ArrayList;
import java.util.List;
import java.util.function.IntConsumer;

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
     * Generation with a cache exactly large enough for the prompt plus the new tokens (no sliding window).
     *
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
        return generate(manager, promptIds, maxGenTokens, eosTokenId, temperature, topK, topP, repetitionPenalty,
                promptIds.length + Math.max(maxGenTokens, 0), null);
    }

    /**
     * Generation with a sliding context window, for texts longer than the length the model was trained on.
     * <p>
     * The cache holds at most {@code contextWindow} tokens. When it is full, the newest half is kept and its cache is
     * rebuilt from position 0 (a context shift). Position 0 restarts, which is exactly what a training window cut from
     * the middle of a long text looks like, so the model never sees positions it was not trained on. Use the training
     * sequence length as the window.
     *
     * @param contextWindow maximum number of tokens in the cache, at least 2
     * @param onToken       called with every generated token as soon as it is sampled (EOS included); may be null
     */
    public List<Integer> generate(NDManager manager,
                                  int[] promptIds,
                                  int maxGenTokens,
                                  int eosTokenId,
                                  float temperature,
                                  int topK,
                                  float topP,
                                  float repetitionPenalty,
                                  int contextWindow,
                                  IntConsumer onToken) {
        if (promptIds.length == 0) {
            throw new IllegalArgumentException("The prompt needs at least one token");
        }
        if (contextWindow < 2) {
            throw new IllegalArgumentException("contextWindow must be at least 2 but was " + contextWindow);
        }
        List<Integer> generated = new ArrayList<>();
        if (maxGenTokens < 1) {
            return generated;
        }

        List<KVCacheBuffer> buffers = model.createLayerBuffers(manager, 1, contextWindow);
        NDArray[] rope = RopeEmbeddings.computeCosSin(manager, contextWindow, model.getHeadDim());

        // The cache holds the most recent tokens; one slot always stays free for the token that is fed next.
        List<Integer> cached = new ArrayList<>();
        for (int i = Math.max(0, promptIds.length - (contextWindow - 1)); i < promptIds.length; i++) {
            cached.add(promptIds[i]);
        }

        // 1. Prefill: the whole (possibly truncated) prompt in one pass
        int next;
        try (NDManager step = manager.newSubManager()) {
            NDArray lastRow = runTokens(step, buffers, rope, toArray(cached), 0);
            next = sampleNextToken(lastRow, generated, temperature, topK, topP, repetitionPenalty);
        }
        generated.add(next);
        if (onToken != null) {
            onToken.accept(next);
        }

        // 2. Decode: feed the token that was just sampled, one position at a time
        int keep = contextWindow / 2;
        while (next != eosTokenId && generated.size() < maxGenTokens) {
            if (cached.size() == contextWindow) {
                cached = new ArrayList<>(cached.subList(cached.size() - keep, cached.size()));
                for (KVCacheBuffer buffer : buffers) {
                    buffer.reset();
                }
                try (NDManager step = manager.newSubManager()) {
                    runTokens(step, buffers, rope, toArray(cached), 0);
                }
            }
            try (NDManager step = manager.newSubManager()) {
                NDArray lastRow = runTokens(step, buffers, rope, new int[]{next}, cached.size());
                cached.add(next);
                next = sampleNextToken(lastRow, generated, temperature, topK, topP, repetitionPenalty);
            }
            generated.add(next);
            if (onToken != null) {
                onToken.accept(next);
            }
        }
        return generated;
    }

    /** Feeds {@code ids} at absolute positions startPosition.. into the caches and returns the logits of the last one. */
    private NDArray runTokens(NDManager step, List<KVCacheBuffer> buffers, NDArray[] rope, int[] ids, int startPosition) {
        NDArray hidden = model.embed(step.create(ids));
        NDArray[] rows = RopeEmbeddings.slice(rope[0], rope[1], startPosition, ids.length);
        rows[0].attach(step);
        rows[1].attach(step);
        NDArray mask = CausalMaskGenerator.createCachedMask(step, ids.length, startPosition + ids.length);
        NDArray logits = model.forwardWithBuffer(step, hidden, mask, rows[0], rows[1], buffers);
        return logits.get((long) (ids.length - 1));
    }

    private static int[] toArray(List<Integer> values) {
        int[] out = new int[values.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = values.get(i);
        }
        return out;
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
