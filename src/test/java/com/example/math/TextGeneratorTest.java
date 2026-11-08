package com.generation;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.transformer.BlockStackLM;
import com.transformer.CausalMaskGenerator;
import com.generation.Sampler;
import com.generation.TextGenerator;
import com.transformer.RopeEmbeddings;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

public class TextGeneratorTest {

    @Test
    public void testAutoregressiveGenerationLoop() {
        try (NDManager manager = NDManager.newBaseManager()) {
            int numLayers = 2;
            int dModel = 16;
            int numHeads = 2;
            int headDim = dModel / numHeads;
            int dFF = 32;
            int vocabSize = 20;
            int promptLen = 4;
            int maxGenTokens = 5;
            int maxSeqLen = promptLen + maxGenTokens;
            float eps = 1e-5f;
            int eosTokenId = 18; // Specific EOS token for stop condition test

            // 1. Instantiate Model, Sampler, and TextGenerator
            BlockStackLM model = new BlockStackLM(manager, numLayers, dModel, numHeads, dFF, vocabSize, eps);
            Sampler sampler = new Sampler();
            TextGenerator generator = new TextGenerator(model, sampler);

            // 2. Prepare Mock Input Data & Pre-computed RoPE Frequencies
            NDArray promptEmbeddings = manager.randomNormal(0.0f, 1.0f, new Shape(promptLen, dModel), DataType.FLOAT32);

            // Compute static RoPE frequency matrices for max sequence length
            NDArray[] ropePair = RopeEmbeddings.computeCosSin(manager, maxSeqLen, headDim, 10000.0f);
            NDArray fullCos = ropePair[0];
            NDArray fullSin = ropePair[1];

            // 3. Execute End-to-End Generation Loop
            List<Integer> generatedTokens = generator.generate(
                    manager,
                    promptEmbeddings,
                    fullCos,
                    fullSin,
                    maxGenTokens,
                    eosTokenId,
                    0.8f,    // Temperature
                    5,       // Top-K
                    0.9f,    // Top-P
                    1.1f     // Repetition Penalty
            );

            // 4. Assertions
            Assertions.assertNotNull(generatedTokens, "Generated token list should not be null");
            Assertions.assertFalse(generatedTokens.isEmpty(), "Generated token list should contain at least one token");
            Assertions.assertTrue(generatedTokens.size() <= maxGenTokens, 
                    "Generated token count should not exceed maxGenTokens (" + maxGenTokens + ")");

            System.out.println("Generated Token IDs: " + generatedTokens);
            System.out.println("Total Tokens Generated: " + generatedTokens.size());
            System.out.println("SUCCESS: TextGenerator end-to-end loop verified.");
        }
    }

    @Test
    public void testEarlyEosTermination() {
        try (NDManager manager = NDManager.newBaseManager()) {
            int numLayers = 1;
            int dModel = 8;
            int numHeads = 2;
            int headDim = dModel / numHeads;
            int dFF = 16;
            int vocabSize = 10;
            int promptLen = 2;
            int maxGenTokens = 10;
            int maxSeqLen = promptLen + maxGenTokens;
            float eps = 1e-5f;

            BlockStackLM model = new BlockStackLM(manager, numLayers, dModel, numHeads, dFF, vocabSize, eps);
            
            Sampler greedySampler = new Sampler() {
                @Override
                public int sampleMultinomial(NDArray probabilities) {
                    return (int) probabilities.argMax().getLong();
                }
            };
            TextGenerator generator = new TextGenerator(model, greedySampler);

            NDArray promptEmbeddings = manager.randomNormal(0.0f, 1.0f, new Shape(promptLen, dModel), DataType.FLOAT32);
            NDArray[] ropePair = RopeEmbeddings.computeCosSin(manager, maxSeqLen, headDim, 10000.0f);

            // Generate valid causal mask for multi-token prompt evaluation
            NDArray promptMask = CausalMaskGenerator.createCachedMask(manager, promptLen, promptLen);
            NDArray promptCos = ropePair[0].get("0:" + promptLen);
            NDArray promptSin = ropePair[1].get("0:" + promptLen);

            int eosTokenId = (int) model.forward(manager, promptEmbeddings, promptMask, promptCos, promptSin)
                    .get((promptLen - 1) + ":")
                    .argMax()
                    .getLong();

            List<Integer> generatedTokens = generator.generate(
                    manager, promptEmbeddings, ropePair[0], ropePair[1],
                    maxGenTokens, eosTokenId, 1.0f, 0, 1.0f, 1.0f
            );

            Assertions.assertEquals(1, generatedTokens.size(), "Generator should terminate immediately on EOS token");
            Assertions.assertEquals(eosTokenId, generatedTokens.get(0).intValue(), "First token generated should match EOS ID");
            System.out.println("SUCCESS: Early EOS termination verified.");
        }
    }
}