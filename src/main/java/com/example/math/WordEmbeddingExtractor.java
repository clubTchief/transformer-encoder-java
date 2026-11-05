package com.example.math;

import ai.djl.inference.Predictor;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;

public class WordEmbeddingExtractor {

    public static void main(String[] args) throws Exception {
        String sentence1 = "The bank of the river was muddy.";
        String sentence2 = "I deposited money at the bank.";
        String sentence3 = "The river shoreline had soft soil.";

        Criteria<String, float[]> criteria = Criteria.builder()
                .setTypes(String.class, float[].class)
                .optModelUrls("djl://ai.djl.huggingface.pytorch/sentence-transformers/all-MiniLM-L6-v2")
                .optEngine("PyTorch")
                .optArgument("normalize", "true") // Normalizing allows direct dot product as cosine similarity
                .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                .build();

        try (ZooModel<String, float[]> model = criteria.loadModel();
             Predictor<String, float[]> predictor = model.newPredictor()) {

            float[] vec1 = predictor.predict(sentence1);
            float[] vec2 = predictor.predict(sentence2);
            float[] vec3 = predictor.predict(sentence3);

            // Compute cosine similarities using your BilinearProduct class from earlier
            double simRiverVsFinance = BilinearProduct.dotProduct(toDoubleArray(vec1), toDoubleArray(vec2));
            double simRiverVsShoreline = BilinearProduct.dotProduct(toDoubleArray(vec1), toDoubleArray(vec3));

            System.out.printf("Similarity (River bank vs. Financial bank): %.4f%n", simRiverVsFinance);
            System.out.printf("Similarity (River bank vs. River shoreline): %.4f%n", simRiverVsShoreline);
        }
    }

    private static double[] toDoubleArray(float[] floatArray) {
        double[] doubleArray = new double[floatArray.length];
        for (int i = 0; i < floatArray.length; i++) {
            doubleArray[i] = floatArray[i];
        }
        return doubleArray;
    }
}