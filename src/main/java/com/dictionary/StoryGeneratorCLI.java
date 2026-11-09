package com.dictionary;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.generation.Sampler;
import com.generation.TextGenerator;
import com.transformer.BlockStackLM;
import com.transformer.ModelSerializer;
import com.transformer.RopeEmbeddings;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class StoryGeneratorCLI {

    public static void main(String[] args) {
        String modelPath = "models/story_model.bin";
        File file = new File(modelPath);

        if (!file.exists()) {
            System.err.println("Model binary not found at: " + modelPath);
            System.err.println("Please run MultiBatchTrainer first to generate the checkpoint.");
            return;
        }

        // Reconstruct corpus & vocabulary used during training
        List<String> corpus = Arrays.asList(
            "Once upon a time in a distant realm lived a brave knight .",
            "The knight found a mysterious glowing sword inside a dark cave .",
            "A powerful dragon protected the ancient castle in the realm .",
            "The brave knight defended the realm and restored peace to all ."
        );

        Vocabulary vocab = new Vocabulary(corpus);
        int seqLen = 12;
        int dModel = 64;
        int numHeads = 4;
        int headDim = dModel / numHeads;
        int numLayers = 2;
        int dFF = 256;
        float eps = 1e-5f;

        System.out.println("================================================================================");
        System.out.println("          >>> TRANSFORMER STORY GENERATOR INTERACTIVE CLI <<<                  ");
        System.out.println("================================================================================");
        System.out.println("Loading saved parameters from: " + modelPath);

        try (NDManager manager = NDManager.newBaseManager()) {

            // 1. Instantiate Model Shell & Embedding Matrix
            BlockStackLM model = new BlockStackLM(manager, numLayers, dModel, numHeads, dFF, vocab.size(), eps);
            NDArray embeddingTable = manager.zeros(new Shape(vocab.size(), dModel), DataType.FLOAT32);

            // Reconstruct total parameter target list (32 block params + 1 embedding matrix)
            List<NDArray> targetParams = new ArrayList<>(model.params());
            targetParams.add(embeddingTable);

            // 2. Deserialize Trained Weights directly into target parameter tensors
            ModelSerializer.loadWeights(manager, modelPath, targetParams);
            System.out.println("Successfully restored " + targetParams.size() + " parameter tensors into engine memory.");
            System.out.println("--------------------------------------------------------------------------------");

            Sampler sampler = new Sampler();
            TextGenerator generator = new TextGenerator(model, sampler);

            // Pre-compute RoPE tables for up to max token sequence length (seqLen + maxNewTokens)
            int maxSeqLen = seqLen + 20;
            NDArray[] ropePair = RopeEmbeddings.computeCosSin(manager, maxSeqLen, headDim, 10000.0f);

            Scanner scanner = new Scanner(System.in);
            System.out.println("Type your prompt words (separated by spaces), or 'exit' to quit.");

            while (true) {
                System.out.print("\nPrompt > ");
                String input = scanner.nextLine().trim();

                if (input.equalsIgnoreCase("exit") || input.equalsIgnoreCase("quit")) {
                    System.out.println("Exiting story generator CLI. Goodbye!");
                    break;
                }

                if (input.isEmpty()) {
                    continue;
                }

                model.resetCache(); // Reset model state for each new prompt

                // 3. Tokenize input string against Vocabulary
                String[] tokens = input.toLowerCase().split("\\s+");
                List<Integer> promptIds = new ArrayList<>();
                for (String token : tokens) {
                    promptIds.add(vocab.getId(token));
                }

                // Convert token IDs to dense embedding matrix [promptLen, dModel]
                int promptLen = promptIds.size();
                float[] embedFlat = embeddingTable.toFloatArray();
                float[] promptEmbedData = new float[promptLen * dModel];

                for (int t = 0; t < promptLen; t++) {
                    int tokenId = promptIds.get(t);
                    System.arraycopy(embedFlat, tokenId * dModel, promptEmbedData, t * dModel, dModel);
                }

                NDArray promptEmbeddings = manager.create(promptEmbedData).reshape(new Shape(promptLen, dModel));

                // 4. Run Autoregressive Inference Generation
                int maxGenTokens = 15;
                int eosTokenId = vocab.getId("<EOS>"); // Safe lookup for EOS ID

                List<Integer> generatedTokenIds = generator.generate(
                        manager,
                        promptEmbeddings,
                        ropePair[0],
                        ropePair[1],
                        maxGenTokens,
                        eosTokenId,
                        0.7f,   // Temperature
                        3,      // Top-K
                        0.9f,   // Top-P
                        1.1f    // Repetition Penalty
                );

                // 5. Decode generated IDs back to text
                StringBuilder resultStory = new StringBuilder(input);
                for (int id : generatedTokenIds) {
                    resultStory.append(" ").append(vocab.getToken(id));
                }

                System.out.println("Generated Story: " + resultStory);
            }

        } catch (IOException e) {
            System.err.println("Failed to read checkpoint: " + e.getMessage());
        }
    }
}