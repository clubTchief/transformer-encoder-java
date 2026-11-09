package com.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.dictionary.StoryModelSpec;
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
        String checkpointPath = StoryModelSpec.CHECKPOINT_PATH;
        File modelFile = new File(checkpointPath);

        if (!modelFile.exists()) {
            System.err.println("[ERROR] Checkpoint file not found at: " + checkpointPath);
            System.err.println("[ERROR] Run MultiBatchTrainer first to generate model weights.");
            return;
        }

        // Training corpus used to dynamically rebuild identical vocabulary mapping
        List<String> corpus = StoryModelSpec.CORPUS;

        Vocabulary vocab = new Vocabulary(corpus);
        int vocabSize = vocab.size();

        try (NDManager manager = NDManager.newBaseManager()) {

            // Sizes come from StoryModelSpec, the same definition the trainer and the CLI use
            BlockStackLM model = new BlockStackLM(manager, StoryModelSpec.config(vocabSize));

            // The embedding table is part of the model, so params() is the whole checkpoint
            List<NDArray> fullTrainableParams = model.params();

            System.out.println("================================================================================");
            System.out.println("          >>> LOADING PRODUCTION STORY GENERATOR PIPELINE <<<                   ");
            System.out.println("================================================================================");

            try {
                ModelSerializer.loadWeights(manager, checkpointPath, fullTrainableParams);
            } catch (IllegalArgumentException e) {
                System.err.println("[ERROR] The checkpoint does not match this model: " + e.getMessage());
                System.err.println("[ERROR] Retrain with MultiBatchTrainer (checkpoints from older versions do not fit).");
                return;
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