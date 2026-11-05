package com.example.math;

import ai.djl.inference.Predictor;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;

public class SentenceEmbeddingsWithPositionalEncoding {

    public static void main(String[] args) throws Exception {
        String sentence = "The bank of the river was muddy.";

        // 1. Fetch semantic sentence embeddings via DJL HuggingFace Model Zoo
        Criteria<String, float[]> criteria = Criteria.builder()
                .setTypes(String.class, float[].class)
                .optModelUrls("djl://ai.djl.huggingface.pytorch/sentence-transformers/all-MiniLM-L6-v2")
                .optEngine("PyTorch")
                .optArgument("normalize", "false")
                .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                .build();

        try (ZooModel<String, float[]> model = criteria.loadModel();
             Predictor<String, float[]> predictor = model.newPredictor();
             NDManager manager = NDManager.newBaseManager()) {

            // Step A: Extract raw semantic embedding vector
            float[] rawVector = predictor.predict(sentence);
            NDArray wordEmbedding = manager.create(rawVector);

            int dModel = (int) wordEmbedding.getShape().get(0); // Dimension (384 for MiniLM)
            int targetPosition = 1;                             // Position of target word "bank" (index 1)

            // Step B: Generate sinusoidal positional encoding vector for position 1
            NDArray positionalVector = generatePositionalVector(manager, targetPosition, dModel);

            // Step C: Combine Semantic Embedding + Positional Encoding (X = E + P)
            NDArray finalInputVector = wordEmbedding.add(positionalVector);

            System.out.println("--- Vector Transformation Breakdown ---");
            System.out.println("Word: 'bank' at position " + targetPosition);
            System.out.println("Embedding Dimension (d_model): " + dModel);
            System.out.println("\nRaw Embedding Shape:       " + wordEmbedding.getShape());
            System.out.println("Positional Encoding Shape: " + positionalVector.getShape());
            System.out.println("Combined Vector (X) Shape: " + finalInputVector.getShape());

            System.out.println("\nFirst 5 components of Combined Vector (X):");
            float[] combinedArray = finalInputVector.toFloatArray();
            System.out.printf("[%.4f, %.4f, %.4f, %.4f, %.4f]%n",
                    combinedArray[0], combinedArray[1], combinedArray[2], combinedArray[3], combinedArray[4]);
        }
    }

    /**
     * Generates a 1D Sinusoidal Positional Encoding vector for a given token position.
     * PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
     * PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
     */
    public static NDArray generatePositionalVector(NDManager manager, int pos, int dModel) {
        float[] pe = new float[dModel];
        for (int i = 0; i < dModel / 2; i++) {
            double denominator = Math.pow(10000.0, (2.0 * i) / dModel);
            pe[2 * i]     = (float) Math.sin(pos / denominator);
            pe[2 * i + 1] = (float) Math.cos(pos / denominator);
        }
        return manager.create(pe);
    }
}