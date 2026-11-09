package com.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.dictionary.Vocabulary;
import com.generation.Sampler;
import com.generation.TextGenerator;

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
            BlockStackLM model = new BlockStackLM(manager, new LMConfig(numLayers, dModel, numHeads, dFF, vocabSize, eps));

            // The embedding table is part of the model, so params() is the whole checkpoint
            List<NDArray> fullTrainableParams = model.params();

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

            // Sampling hyper-parameters
            float temperature = 0.7f;
            int topK = 5;
            float topP = 0.9f;

            // KV-cached generation: the prompt is processed once, then every sampled token is embedded and fed back in
            int[] promptIds = generatedIds.stream().mapToInt(Integer::intValue).toArray();
            TextGenerator generator = new TextGenerator(model, new Sampler());
            List<Integer> newIds = generator.generate(manager, promptIds, maxTokensToGenerate,
                    vocab.getId(Vocabulary.EOS_TOKEN), temperature, topK, topP, 1.0f);

            for (int id : newIds) {
                String word = vocab.getToken(id);
                // Stop if the generator hits EOS or PAD tokens
                if (Vocabulary.EOS_TOKEN.equals(word) || Vocabulary.PAD_TOKEN.equals(word)) {
                    break;
                }
                System.out.print(word + " ");
            }

            System.out.println("\n\n=== GENERATION COMPLETE ===");
            System.out.println("================================================================================\n");
        }
    }
}