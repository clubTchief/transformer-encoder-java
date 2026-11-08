package com.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.dictionary.Vocabulary;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class StoryGeneratorPipeline {

    public static void main(String[] args) {
        String checkpointPath = "models/story_model.bin";
        File modelFile = new File(checkpointPath);

        if (!modelFile.exists()) {
            System.err.println("[ERROR] Checkpoint file not found at: " + checkpointPath);
            System.err.println("[ERROR] Run MultiBatchTrainer first to generate model weights.");
            return;
        }

        // Training corpus used to dynamically rebuild identical vocabulary mapping
        List<String> corpus = Arrays.asList(
            "Once upon a time in a distant realm lived a brave knight .",
            "The knight found a mysterious glowing sword inside a dark cave .",
            "A powerful dragon protected the ancient castle in the realm .",
            "The brave knight defended the realm and restored peace to all ."
        );

        Vocabulary vocab = new Vocabulary(corpus);
        int vocabSize = vocab.size();

        try (NDManager manager = NDManager.newBaseManager()) {

            // Model Hyperparameters
            int dModel = 64;
            int numHeads = 4;
            int numLayers = 2;
            int dFF = 256;
            float eps = 1e-5f;

            // Instantiate architecture and parameter embeddings
            BlockStackLM model = new BlockStackLM(manager, dModel, numHeads, numLayers, dFF, vocabSize, eps);
            NDArray embeddingTable = manager.zeros(new Shape(vocabSize, dModel), DataType.FLOAT32);

            List<NDArray> fullTrainableParams = new ArrayList<>(model.params());
            fullTrainableParams.add(embeddingTable);

            System.out.println("================================================================================");
            System.out.println("          >>> LOADING PRODUCTION STORY GENERATOR PIPELINE <<<                   ");
            System.out.println("================================================================================");

            try {
                ModelSerializer.loadWeights(manager, checkpointPath, fullTrainableParams);
            } catch (IOException e) {
                System.err.println("[ERROR] Failed to load checkpoint weights: " + e.getMessage());
                return;
            }

            // Prompt definition
            String promptText = "Once upon a time";
            String[] promptTokens = Vocabulary.tokenize(promptText);

            List<Integer> generatedIds = new ArrayList<>();
            generatedIds.add(vocab.getId(Vocabulary.BOS_TOKEN));

            for (String token : promptTokens) {
                int id = vocab.getId(token);
                if (id != vocab.getId(Vocabulary.UNK_TOKEN)) {
                    generatedIds.add(id);
                }
            }

            System.out.println("--------------------------------------------------------------------------------");
            System.out.println("=== STARTING AUTOREGRESSIVE STORY GENERATION ===");
            System.out.println("Prompt: " + promptText);
            System.out.println("--------------------------------------------------------------------------------");

            // Print prompt tokens
            for (int i = 1; i < generatedIds.size(); i++) {
                System.out.print(vocab.getToken(generatedIds.get(i)) + " ");
            }

            int maxTokensToGenerate = 25;
            int targetLength = generatedIds.size() + maxTokensToGenerate;

            // Sampling hyper-parameters
            float temperature = 0.7f;
            int topK = 5;
            float topP = 0.9f;

            while (generatedIds.size() < targetLength) {
                int currentLen = generatedIds.size();

                // 1. Build 2D input matrix [currentLen, dModel] from embeddings
                NDArray genInputX = manager.create(new Shape(currentLen, dModel), DataType.FLOAT32);
                for (int i = 0; i < currentLen; i++) {
                    NDArray tokenVec = embeddingTable.get(generatedIds.get(i));
                    genInputX.set(new ai.djl.ndarray.index.NDIndex("{}:{}", i, i + 1), tokenVec);
                }

                // 2. Generate causal mask and compute forward logits
                NDArray genMask4D = CausalMaskGenerator.createCausalMask(manager, currentLen);
                NDArray genMask = genMask4D.squeeze(0).squeeze(0);
                NDArray logits = model.forward(manager, genInputX, genMask);

                // 3. Extract logit projection for the last sequence position: shape [vocabSize]
                NDArray lastLogits = logits.get((long) (currentLen - 1));
                float[] logitVals = lastLogits.toFloatArray();

                // 4. Sample next token using Temperature + Top-P / Top-K sampling
                int nextTokenId = SamplingUtils.sampleToken(logitVals, temperature, topK, topP);
                String word = vocab.getToken(nextTokenId);

                // Stop if generator hits EOS or PAD tokens
                if (Vocabulary.EOS_TOKEN.equals(word) || Vocabulary.PAD_TOKEN.equals(word)) {
                    break;
                }

                System.out.print(word + " ");
                generatedIds.add(nextTokenId);
            }

            System.out.println("\n\n=== GENERATION COMPLETE ===");
            System.out.println("================================================================================\n");
        }
    }
}