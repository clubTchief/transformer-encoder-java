package com.encoder;

import ai.djl.inference.Predictor;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;

public class IntegratedAttentionPipeline {

    public static void main(String[] args) throws Exception {
        printHeader();

        String sentence = "The bank of the river was muddy";
        String[] tokens = sentence.split(" ");
        int seqLen = tokens.length;
        int numHeads = 8; // Multi-Head Attention Configuration

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

            // =========================================================================
            // STEP 1: SEMANTIC WORD EMBEDDINGS (E)
            // =========================================================================
            float[][] rawEmbeddings = new float[seqLen][];
            for (int i = 0; i < seqLen; i++) {
                rawEmbeddings[i] = predictor.predict(tokens[i]);
            }
            NDArray E = manager.create(rawEmbeddings);
            int dModel = (int) E.getShape().get(1);

            explainStep1(tokens, E, rawEmbeddings, dModel);

            // =========================================================================
            // STEP 2: POSITIONAL ENCODING (P) & COMPOSITION (X = E + P)
            // =========================================================================
            NDArray P = SentenceEmbeddingsWithPositionalEncoding.generatePositionalMatrix(manager, seqLen, dModel);
            NDArray X = E.add(P);

            explainStep2(tokens, P, X);

            // =========================================================================
            // STEP 3: MULTI-HEAD ATTENTION (MHA) PIPELINE
            // =========================================================================
            int dK = dModel / numHeads; // 384 / 8 = 48
            int dV = dK;                 // 48

            float stddevK = (float) Math.sqrt(2.0 / (dModel + dK));
            float stddevO = (float) Math.sqrt(2.0 / (dModel + dModel));

            // Weights: W_Q, W_K, W_V in [d_model, d_model], W_O in [d_model, d_model]
            NDArray W_Q = manager.randomNormal(0.0f, stddevK, new Shape(dModel, dModel), DataType.FLOAT32);
            NDArray W_K = manager.randomNormal(0.0f, stddevK, new Shape(dModel, dModel), DataType.FLOAT32);
            NDArray W_V = manager.randomNormal(0.0f, stddevK, new Shape(dModel, dModel), DataType.FLOAT32);
            NDArray W_O = manager.randomNormal(0.0f, stddevO, new Shape(dModel, dModel), DataType.FLOAT32);

            // 1. Projection: [N, d_model] * [d_model, d_model] -> [N, d_model]
            NDArray Q = X.matMul(W_Q);
            NDArray K = X.matMul(W_K);
            NDArray V = X.matMul(W_V);

            // 2. Reshape & Transpose to Split Heads: [N, d_model] -> [N, h, d_k] -> [h, N, d_k]
            NDArray Q_split = Q.reshape(new Shape(seqLen, numHeads, dK)).transpose(1, 0, 2);
            NDArray K_split = K.reshape(new Shape(seqLen, numHeads, dK)).transpose(1, 0, 2);
            NDArray V_split = V.reshape(new Shape(seqLen, numHeads, dV)).transpose(1, 0, 2);

            // 3. Batched Matrix Multiplication per head: [h, N, d_k] * [h, d_k, N] -> [h, N, N]
            NDArray K_transposed = K_split.transpose(0, 2, 1);
            NDArray djlScores = Q_split.matMul(K_transposed).div(Math.sqrt(dK));
            NDArray djlAttentionWeights = djlScores.softmax(-1); // [h, N, N]

            // 4. Value Aggregation per head: [h, N, N] * [h, N, d_v] -> [h, N, d_v]
            NDArray headOutputs = djlAttentionWeights.matMul(V_split);

            // 5. Re-combine / Concatenate Heads: [h, N, d_v] -> [N, h, d_v] -> [N, d_model]
            NDArray concatOutputs = headOutputs.transpose(1, 0, 2).reshape(new Shape(seqLen, dModel));

            // 6. Output Projection: [N, d_model] * [d_model, d_model] -> [N, d_model]
            NDArray mhaOutput = concatOutputs.matMul(W_O);

            explainMultiHeadStep(tokens, numHeads, dModel, dK, Q_split, djlAttentionWeights, mhaOutput);
        }
    }

    // =========================================================================
    // EXPLANATION LOGGING HELPERS
    // =========================================================================

    private static void printHeader() {
        System.out.println("==========================================================================");
        System.out.println("         TRANSFORMER MULTI-HEAD SELF-ATTENTION (MHA) EXPLAINER            ");
        System.out.println("==========================================================================");
    }

    private static void explainStep1(String[] tokens, NDArray E, float[][] rawEmbeddings, int dModel) {
        System.out.println("\n[STEP 1] Semantic Token Embeddings (E)");
        System.out.println("--------------------------------------------------------------------------");
        System.out.println("Explanation : Each word is converted into a high-dimensional vector via Hugging Face.");
        System.out.println("Formula     : E = [ Lookup(t_0), Lookup(t_1), ..., Lookup(t_N) ]");
        System.out.println("Tokens      : " + String.join(", ", tokens));
        System.out.println("Output Shape: " + E.getShape() + " (Sequence Length N = " + tokens.length + ", Embedding Dim d_model = " + dModel + ")\n");
    }

    private static void explainStep2(String[] tokens, NDArray P, NDArray X) {
        System.out.println("\n[STEP 2] Sinusoidal Positional Encoding & Composition (X = E + P)");
        System.out.println("--------------------------------------------------------------------------");
        System.out.println("Explanation : We add a sinusoidal position matrix (P) to the semantic embeddings (E).");
        System.out.println("Formula     : X = E + P");
        System.out.println("Input X Shape: " + X.getShape() + " (Combined spatial + semantic input)\n");
    }

    private static void explainMultiHeadStep(String[] tokens, int numHeads, int dModel, int dK,
                                             NDArray Q_split, NDArray attentionWeights, NDArray mhaOutput) {
        System.out.println("\n[STEP 3] Multi-Head Attention (MHA) Execution & Tensor Transformations");
        System.out.println("--------------------------------------------------------------------------");
        System.out.println("Configuration : Number of Heads h = " + numHeads + " | Subspace Dimension d_k = " + dModel + " / " + numHeads + " = " + dK);
        System.out.println("\n--- Tensor Dimensional Tracking ---");
        System.out.printf("%-35s | %-20s%n", "Pipeline Phase", "Tensor Shape [Dimensions]");
        System.out.println("-".repeat(60));
        System.out.printf("%-35s | %-20s%n", "1. Head Splitting (Q, K, V)", Q_split.getShape().toString());
        System.out.printf("%-35s | %-20s%n", "2. Batched Softmax Weights (A)", attentionWeights.getShape().toString());
        System.out.printf("%-35s | %-20s%n", "3. Concatenated Heads (Concat)", "[7, 384]");
        System.out.printf("%-35s | %-20s%n", "4. Final Projected Output (MHA)", mhaOutput.getShape().toString());
        System.out.println("-".repeat(60));

        // Display sample weights from Head 0 and Head 1 to demonstrate representation diversity
        System.out.println("\n--- Attention Subspace Diversity (Comparison: Head 0 vs. Head 1) ---");
        float[] flatWeights = attentionWeights.toFloatArray();
        int seqLen = tokens.length;

        System.out.println("Head 0 Top Weight pair for 'bank':");
        printHeadTopWeight(tokens, flatWeights, 0, seqLen, 1); // Row 1 = 'bank'

        System.out.println("Head 1 Top Weight pair for 'bank':");
        printHeadTopWeight(tokens, flatWeights, 1, seqLen, 1);

        System.out.println("\n--- Final Multi-Head Context Vectors (First 4 Dimensions Shown) ---");
        float[][] outArray = toFloatArray2D(mhaOutput);
        System.out.printf("%-10s | %-10s | %-45s%n", "Index", "Word", "Context Vector Slice [d0, d1, d2, d3...]");
        System.out.println("-".repeat(70));

        for (int i = 0; i < tokens.length; i++) {
            float[] vec = outArray[i];
            String snippet = String.format("[%.4f, %.4f, %.4f, %.4f, ...]", vec[0], vec[1], vec[2], vec[3]);
            System.out.printf("%-10d | %-10s | %-45s%n", i, tokens[i], snippet);
        }
        System.out.println("==========================================================================");
    }

    private static void printHeadTopWeight(String[] tokens, float[] flatWeights, int headIdx, int seqLen, int queryRow) {
        int headOffset = headIdx * seqLen * seqLen;
        int rowOffset = headOffset + queryRow * seqLen;

        int maxCol = 0;
        float maxVal = -1.0f;

        for (int j = 0; j < seqLen; j++) {
            float val = flatWeights[rowOffset + j];
            if (val > maxVal) {
                maxVal = val;
                maxCol = j;
            }
        }
        System.out.printf("  -> Head %d: '%s' attends most heavily to '%s' (Weight = %.4f)%n", 
                headIdx, tokens[queryRow], tokens[maxCol], maxVal);
    }

    // =========================================================================
    // INNER CORE MATH HELPERS
    // =========================================================================

    public static class SentenceEmbeddingsWithPositionalEncoding {
        public static NDArray generatePositionalMatrix(NDManager manager, int seqLen, int dModel) {
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
    }

    private static float[][] toFloatArray2D(NDArray array) {
        Shape shape = array.getShape();
        int rows = (int) shape.get(0);
        int cols = (int) shape.get(1);
        float[] flat = array.toFloatArray();
        float[][] matrix = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            System.arraycopy(flat, i * cols, matrix[i], 0, cols);
        }
        return matrix;
    }
}