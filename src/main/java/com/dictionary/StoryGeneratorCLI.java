package com.dictionary;

import ai.djl.ndarray.NDManager;
import com.generation.Sampler;
import com.generation.TextGenerator;
import com.transformer.BlockStackLM;
import com.transformer.ModelSerializer;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Scanner;

/** Interactive story generator: loads the checkpoint written by MultiBatchTrainer and continues what you type. */
public class StoryGeneratorCLI {

    private static final int MAX_NEW_TOKENS = 15;
    private static final float TEMPERATURE = 0.7f;
    private static final int TOP_K = 3;
    private static final float TOP_P = 0.9f;
    private static final float REPETITION_PENALTY = 1.1f;

    public static void main(String[] args) {
        String modelPath = StoryModelSpec.CHECKPOINT_PATH;
        if (!new File(modelPath).exists()) {
            System.err.println("Model binary not found at: " + modelPath);
            System.err.println("Please run MultiBatchTrainer first to generate the checkpoint.");
            return;
        }

        Vocabulary vocab = StoryModelSpec.vocabulary();

        System.out.println("================================================================================");
        System.out.println("          >>> TRANSFORMER STORY GENERATOR INTERACTIVE CLI <<<                  ");
        System.out.println("================================================================================");
        System.out.println("Loading saved parameters from: " + modelPath);

        try (NDManager manager = NDManager.newBaseManager(); Scanner scanner = new Scanner(System.in)) {

            // params() is the whole model, embedding table included
            BlockStackLM model = new BlockStackLM(manager, StoryModelSpec.config(vocab.size()));
            try {
                ModelSerializer.loadWeights(manager, modelPath, model.params());
            } catch (IllegalArgumentException e) {
                System.err.println("The checkpoint does not match this model: " + e.getMessage());
                System.err.println("Retrain with MultiBatchTrainer (a checkpoint from MultiEpochTrainer or an older version won't fit).");
                return;
            }
            System.out.println("Restored " + model.params().size() + " parameter tensors. Vocabulary size: " + vocab.size());
            System.out.println("--------------------------------------------------------------------------------");

            TextGenerator generator = new TextGenerator(model, new Sampler());
            int eosTokenId = vocab.getId(Vocabulary.EOS_TOKEN);

            System.out.println("Type the start of a story (words are case-insensitive, punctuation is fine), or 'exit' to quit.");
            while (true) {
                System.out.print("\nPrompt > ");
                if (!scanner.hasNextLine()) {
                    break; // end of input (Ctrl-D / Ctrl-Z)
                }
                String input = scanner.nextLine().trim();

                if (input.equalsIgnoreCase("exit") || input.equalsIgnoreCase("quit")) {
                    System.out.println("Exiting story generator CLI. Goodbye!");
                    break;
                }
                if (input.isEmpty()) {
                    continue;
                }

                PromptCodec.Encoded prompt = PromptCodec.encode(vocab, input);
                if (!prompt.unknownWords().isEmpty()) {
                    System.out.println("(ignored words that are not in the vocabulary: " + prompt.unknownWords() + ")");
                }
                if (prompt.ids().length == 1) { // only BOS: nothing usable was typed
                    System.out.println("None of those words are known. Try words from the training stories, e.g. 'Once upon a time'.");
                    continue;
                }

                List<Integer> generated = generator.generate(manager, prompt.ids(), MAX_NEW_TOKENS, eosTokenId,
                        TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY);

                List<Integer> promptIds = java.util.Arrays.stream(prompt.ids()).boxed().toList();
                System.out.println("Generated Story: " + PromptCodec.decode(vocab, promptIds)
                        + " " + PromptCodec.decode(vocab, generated));
            }

        } catch (IOException e) {
            System.err.println("Failed to read checkpoint: " + e.getMessage());
        }
    }
}
