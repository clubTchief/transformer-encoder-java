package com.example.math;

import ai.djl.inference.Predictor;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;

public class TransformerAttentionPipeline {

    public static void main(String[] args) throws Exception {
        String[] words = {"The", "bank", "of", "the", "river"};
        int seqLen = words.length;

        // 1. Load HuggingFace model criteria via DJL
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

            // Step 1: Extract Semantic Embeddings for each word in context
            float[] sampleVec = predictor.predict("The bank of the river");
            int dModel = sampleVec.length; // 384 dimensions

            // Collect vectors into a 2D float array
            float[][] embeddingsArray = new float[seqLen][dModel];
            for (int pos = 0; pos < seqLen; pos++) {
                embeddingsArray[pos] = predictor.predict(words[pos]);
            }
            NDArray semanticEmbeddings = manager.create(embeddingsArray);

            // Step 2: Add Positional Encodings -> X = Embeddings + PE
            NDArray positionalEncodings = generatePositionalMatrix(manager, seqLen, dModel);
            NDArray X = semanticEmbeddings.add(positionalEncodings);

            System.out.println("=== Stage 1: Input Matrix X (Embeddings + Positional Encoding) ===");
            System.out.println("Shape of X: " + X.getShape()); // [5, 384]

            // Step 3: Project X into Q, K, V Matrices
            int dK = 64; // Projection dimension for Queries & Keys
            int dV = 64; // Projection dimension for Values

            // Random weights representing learned projections (W_Q, W_K, W_V)
            NDArray W_Q = manager.randomNormal(0.0f, 1.0f, new Shape(dModel, dK), DataType.FLOAT32);
            NDArray W_K = manager.randomNormal(0.0f, 1.0f, new Shape(dModel, dK), DataType.FLOAT32);
            NDArray W_V = manager.randomNormal(0.0f, 1.0f, new Shape(dModel, dV), DataType.FLOAT32);

            NDArray Q = X.matMul(W_Q); // Shape: [5, 64]
            NDArray K = X.matMul(W_K); // Shape: [5, 64]
            NDArray V = X.matMul(W_V); // Shape: [5, 64]

            System.out.println("\n=== Stage 2: Q, K, V Projections ===");
            System.out.println("Query Matrix Q shape: " + Q.getShape());
            System.out.println("Key Matrix K shape:   " + K.getShape());
            System.out.println("Value Matrix V shape: " + V.getShape());

            // Step 4: Compute Scaled Dot-Product Self-Attention
            NDArray attentionOutput = computeScaledDotProductAttention(manager, Q, K, V, dK);

            System.out.println("\n=== Stage 3: Self-Attention Contextual Output ===");
            System.out.println("Final Output Matrix shape: " + attentionOutput.getShape()); // [5, 64]
        }
    }

    /**
     * Generates a 2D Sinusoidal Positional Encoding matrix [seqLen, dModel].
     */
    private static NDArray generatePositionalMatrix(NDManager manager, int seqLen, int dModel) {
        float[][] pe = new float[seqLen][dModel];
        for (int pos = 0; pos < seqLen; pos++) {
            for (int i = 0; i < dModel / 2; i++) {
                double denominator = Math.pow(10000.0, (2.0 * i) / dModel);
                pe[pos][2 * i]     = (float) Math.sin(pos / denominator);
                pe[pos][2 * i + 1] = (float) Math.cos(pos / denominator);
            }
        }
        return manager.create(pe);
    }

    /**
     * Computes Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
     */
    private static NDArray computeScaledDotProductAttention(NDManager manager, NDArray Q, NDArray K, NDArray V, int dK) {
        NDArray scores = Q.matMul(K.transpose());                  // [seqLen, seqLen]
        NDArray scaledScores = scores.div(Math.sqrt(dK));          // Scale by sqrt(d_k)
        NDArray attentionWeights = scaledScores.softmax(-1);        // Softmax over target axis
        return attentionWeights.matMul(V);                         // [seqLen, dV]
    }
}