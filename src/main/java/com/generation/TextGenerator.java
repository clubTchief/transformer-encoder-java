package com.generation;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.transformer.BlockStackLM;
import com.transformer.CausalMaskGenerator;
import com.transformer.KVCacheBuffer;

import java.util.ArrayList;
import java.util.List;

public class TextGenerator {

    private final BlockStackLM model;
    private final LogitsProcessor logitsProcessor;
    private final Sampler sampler;

    public TextGenerator(BlockStackLM model, Sampler sampler) {
        this.model = model;
        this.logitsProcessor = new LogitsProcessor();
        this.sampler = sampler;
    }

    /**
     * Executes autoregressive token generation given prompt token embeddings.
     *
     * @param manager Sub-manager for scoped tensor allocations
     * @param promptEmbeddings Tensor of shape [promptLen, dModel]
     * @param fullCos Pre-computed RoPE cosine table [maxSeqLen, headDim]
     * @param fullSin Pre-computed RoPE sine table [maxSeqLen, headDim]
     * @param maxGenTokens Maximum new tokens to generate
     * @param eosTokenId Optional end-of-sequence token ID (-1 to disable)
     * @param temperature Sampling temperature
     * @param topK Top-K cutoff
     * @param topP Top-P cumulative threshold
     * @param repetitionPenalty Penalty factor for repeated tokens
     * @return List of generated token IDs (including prompt indices if tracked)
     */
    public List<Integer> generate(NDManager manager,
                                  NDArray promptEmbeddings,
                                  NDArray fullCos,
                                  NDArray fullSin,
                                  int maxGenTokens,
                                  int eosTokenId,
                                  float temperature,
                                  int topK,
                                  float topP,
                                  float repetitionPenalty) {

        int promptLen = (int) promptEmbeddings.getShape().get(0);
        int maxSeqLen = promptLen + maxGenTokens;

        List<KVCacheBuffer> layerBuffers = model.createLayerBuffers(manager, 1, maxSeqLen);
        List<Integer> generatedTokenIds = new ArrayList<>();

        // 1. Prefill Phase (Prompt tokens 0..promptLen)
        NDArray promptCos = fullCos.get("0:" + promptLen);
        NDArray promptSin = fullSin.get("0:" + promptLen);
        NDArray promptMask = CausalMaskGenerator.createCachedMask(manager, promptLen, promptLen);

        NDArray promptLogits = model.forwardWithBuffer(manager, promptEmbeddings, promptMask, promptCos, promptSin, layerBuffers);

        // Extract logits for the last prompt token position
        NDArray lastPromptLogit = promptLogits.get((promptLen - 1) + ":");

        int nextToken = sampleNextToken(lastPromptLogit, generatedTokenIds, temperature, topK, topP, repetitionPenalty);
        generatedTokenIds.add(nextToken);

        if (nextToken == eosTokenId) {
            return generatedTokenIds;
        }

        // 2. Incremental Step Generation Loop
        // Note: For real token generation, provide an Embedding Matrix lookup layer to convert nextToken -> 1xD vector.
        // For testing/demonstration, we use the model's model dimension context.
        for (int step = 0; step < maxGenTokens - 1; step++) {
            int currentPos = promptLen + step;

            // Slice position-specific RoPE frequencies for step position
            NDArray stepCos = fullCos.get(currentPos + ":" + (currentPos + 1));
            NDArray stepSin = fullSin.get(currentPos + ":" + (currentPos + 1));
            NDArray stepMask = CausalMaskGenerator.createCachedMask(manager, 1, currentPos + 1);

            // Fetch embedding vector for nextToken (using dummy projection as placeholder)
            NDArray stepEmbedding = manager.randomNormal(0.0f, 1.0f, new Shape(1, model.getDModel()), DataType.FLOAT32);

            NDArray stepLogits = model.forwardWithBuffer(manager, stepEmbedding, stepMask, stepCos, stepSin, layerBuffers);
            NDArray singleLogit = stepLogits.squeeze(0);

            nextToken = sampleNextToken(singleLogit, generatedTokenIds, temperature, topK, topP, repetitionPenalty);
            generatedTokenIds.add(nextToken);

            if (nextToken == eosTokenId) {
                break;
            }
        }

        return generatedTokenIds;
    }

    private int sampleNextToken(NDArray rawLogits,
                                List<Integer> history,
                                float temperature,
                                int topK,
                                float topP,
                                float repetitionPenalty) {

        NDArray processed = LogitsProcessor.applyRepetitionPenalty(rawLogits, history, repetitionPenalty);
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