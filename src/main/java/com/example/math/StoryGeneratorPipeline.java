package com.example.math;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class StoryGeneratorPipeline {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {

            // Pipeline Hyperparameters
            int dModel = 384;
            int numHeads = 8;
            int numLayers = 4;
            int dFF = 4 * dModel;
            float eps = 1e-5f;
            int maxGenerateTokens = 20;
            float temperature = 0.8f;
            int topK = 5;

            // Toy Vocabulary Mapping (Token ID <-> Word)
            Map<Integer, String> idToToken = new HashMap<>();
            Map<String, Integer> tokenToId = new HashMap<>();
            String[] vocabulary = {
                "<PAD>", "Once", "upon", "a", "time", "in", "a", "distant", 
                "galaxy", "there", "lived", "a", "brave", "knight", "who", 
                "discovered", "a", "mysterious", "glowing", "portal", "."
            };

            for (int i = 0; i < vocabulary.length; i++) {
                idToToken.put(i, vocabulary[i]);
                tokenToId.putIfAbsent(vocabulary[i], i);
            }
            int vocabSize = idToToken.size();

            // Initialize Token Embedding Table [vocabSize, dModel]
            NDArray embeddingTable = manager.randomNormal(0.0f, 0.02f, new Shape(vocabSize, dModel), DataType.FLOAT32);

            // Initialize Decoder Block Stack
            List<DecoderBlock> blocks = new ArrayList<>();
            for (int l = 0; l < numLayers; l++) {
                blocks.add(new DecoderBlock(manager, dModel, numHeads, dFF, eps));
            }

            // Initialize LM Head
            LMHead lmHead = new LMHead(manager, dModel, vocabSize);

            // Seed Prompt: "Once upon a time"
            List<Integer> generatedTokenIds = new ArrayList<>(List.of(
                tokenToId.get("Once"),
                tokenToId.get("upon"),
                tokenToId.get("a"),
                tokenToId.get("time")
            ));

            System.out.println("=== STARTING AUTOREGRESSIVE STORY GENERATION ===");
            System.out.print("Prompt: ");
            for (int id : generatedTokenIds) System.out.print(idToToken.get(id) + " ");
            System.out.println("\n" + "-".repeat(60));

            // Autoregressive Loop
            for (int step = 0; step < maxGenerateTokens; step++) {
                int seqLen = generatedTokenIds.size();

                // 1. Convert Token IDs to Embeddings [seqLen, dModel]
                NDArray currentX = manager.create(new Shape(seqLen, dModel), DataType.FLOAT32);
                for (int i = 0; i < seqLen; i++) {
                    NDArray tokenVector = embeddingTable.get(generatedTokenIds.get(i));
                    currentX.set(new ai.djl.ndarray.index.NDIndex("{}:{}", i, i + 1), tokenVector);
                }

                // 2. Prepare Causal Mask & Pre-compute RoPE
                NDArray causalMask4D = CausalMaskGenerator.createCausalMask(manager, seqLen);
                NDArray causalMask = causalMask4D.squeeze(0).squeeze(0);
                NDArray[] ropeFreqs = ModernizedHierarchicalEncoderPipeline.precomputeRoPEFrequencies(manager, seqLen, dModel / numHeads);

                // 3. Forward Pass through Transformer Layers
                for (DecoderBlock block : blocks) {
                    currentX = block.forward(currentX, causalMask, ropeFreqs[0], ropeFreqs[1]);
                }

                // 4. Project Final Output Token Hidden State to Vocab Logits
                NDArray allLogits = lmHead.forward(currentX); // [seqLen, vocabSize]
                NDArray lastTokenLogits = allLogits.get((long) (seqLen - 1)); // [vocabSize]

                // 5. Sample Next Token
                int nextTokenId = SamplingUtils.sampleNextToken(lastTokenLogits, temperature, topK);
                generatedTokenIds.add(nextTokenId);

                System.out.print(idToToken.get(nextTokenId) + " ");

                // Stop if period token generated
                if (idToToken.get(nextTokenId).equals(".")) break;
            }

            System.out.println("\n\n=== GENERATION COMPLETE ===");
        }
    }
}