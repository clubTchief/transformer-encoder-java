package com.encoder;

import ai.djl.inference.Predictor;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;

public class EncoderBlock {

    public static void main(String[] args) throws Exception {
        printHeader();

        String sentence = "The bank of the river was muddy";
        String[] tokens = sentence.split(" ");
        int seqLen = tokens.length;
        int numHeads = 8;
        float eps = 1e-5f;

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
            int dModel = (int) E.getShape().get(1); // d_model = 384

            explainStep1(tokens, E, rawEmbeddings, dModel);

            // =========================================================================
            // STEP 2: POSITIONAL ENCODING (P) & COMPOSITION (X = E + P)
            // =========================================================================
            NDArray P = generatePositionalMatrix(manager, seqLen, dModel);
            NDArray X = E.add(P);

            explainStep2(tokens, P, X);

            // =========================================================================
            // STEP 3: SUBLAYER 1 - MULTI-HEAD ATTENTION (MHA)
            // =========================================================================
            int dK = dModel / numHeads; // 384 / 8 = 48
            int dV = dK;
            float stddevK = (float) Math.sqrt(2.0 / (dModel + dK));
            float stddevO = (float) Math.sqrt(2.0 / (dModel + dModel));

            NDArray W_Q = manager.randomNormal(0.0f, stddevK, new Shape(dModel, dModel), DataType.FLOAT32);
            NDArray W_K = manager.randomNormal(0.0f, stddevK, new Shape(dModel, dModel), DataType.FLOAT32);
            NDArray W_V = manager.randomNormal(0.0f, stddevK, new Shape(dModel, dModel), DataType.FLOAT32);
            NDArray W_O = manager.randomNormal(0.0f, stddevO, new Shape(dModel, dModel), DataType.FLOAT32);

            NDArray Q = X.matMul(W_Q);
            NDArray K = X.matMul(W_K);
            NDArray V = X.matMul(W_V);

            NDArray Q_split = Q.reshape(new Shape(seqLen, numHeads, dK)).transpose(1, 0, 2);
            NDArray K_split = K.reshape(new Shape(seqLen, numHeads, dK)).transpose(1, 0, 2);
            NDArray V_split = V.reshape(new Shape(seqLen, numHeads, dV)).transpose(1, 0, 2);

            NDArray K_transposed = K_split.transpose(0, 2, 1);
            NDArray djlScores = Q_split.matMul(K_transposed).div(Math.sqrt(dK));
            NDArray djlAttentionWeights = djlScores.softmax(-1);

            NDArray headOutputs = djlAttentionWeights.matMul(V_split);
            NDArray concatOutputs = headOutputs.transpose(1, 0, 2).reshape(new Shape(seqLen, dModel));
            NDArray mhaOutput = concatOutputs.matMul(W_O);

            explainStep3(tokens, numHeads, dModel, dK, Q, K, V, Q_split, djlScores, djlAttentionWeights, mhaOutput);

            // =========================================================================
            // STEP 4: SUBLAYER 1 - RESIDUAL CONNECTION & LAYER NORM (Add & Norm 1)
            // =========================================================================
            NDArray subLayer1Residual = X.add(mhaOutput);
            NDArray subLayer1Output = applyLayerNorm(manager, subLayer1Residual, eps);

            explainStep4(tokens, X, mhaOutput, subLayer1Residual, subLayer1Output);

            // =========================================================================
            // STEP 5: SUBLAYER 2 - POSITION-WISE FEED-FORWARD NETWORK (FFN)
            // =========================================================================
            int dFF = 4 * dModel; // 4 * 384 = 1536
            float stddev1 = (float) Math.sqrt(2.0 / (dModel + dFF));
            float stddev2 = (float) Math.sqrt(2.0 / (dFF + dModel));

            NDArray W_1 = manager.randomNormal(0.0f, stddev1, new Shape(dModel, dFF), DataType.FLOAT32);
            NDArray b_1 = manager.zeros(new Shape(dFF), DataType.FLOAT32);
            NDArray W_2 = manager.randomNormal(0.0f, stddev2, new Shape(dFF, dModel), DataType.FLOAT32);
            NDArray b_2 = manager.zeros(new Shape(dModel), DataType.FLOAT32);

            NDArray linear1 = subLayer1Output.matMul(W_1).add(b_1);
            NDArray ffnHidden = Activation.relu(linear1);
            NDArray ffnOutput = ffnHidden.matMul(W_2).add(b_2);

            explainStep5(tokens, dModel, dFF, linear1, ffnHidden, ffnOutput);

            // =========================================================================
            // STEP 6: SUBLAYER 2 - RESIDUAL CONNECTION & LAYER NORM (Add & Norm 2)
            // =========================================================================
            NDArray subLayer2Residual = subLayer1Output.add(ffnOutput);
            NDArray finalEncoderOutput = applyLayerNorm(manager, subLayer2Residual, eps);

            explainStep6(tokens, X, subLayer1Output, ffnOutput, subLayer2Residual, finalEncoderOutput);
        }
    }

    // =========================================================================
    // LAYER NORMALIZATION & POSITIONAL ENCODING HELPERS
    // =========================================================================

    public static NDArray applyLayerNorm(NDManager manager, NDArray X, float epsilon) {
        Shape shape = X.getShape();
        long dModel = shape.get(1);

        NDArray mean = X.mean(new int[]{-1}, true);
        NDArray centered = X.sub(mean);
        NDArray variance = centered.square().mean(new int[]{-1}, true);

        NDArray stdDev = variance.add(epsilon).sqrt();
        NDArray normalized = centered.div(stdDev);

        NDArray gamma = manager.ones(new Shape(dModel));
        NDArray beta = manager.zeros(new Shape(dModel));

        return normalized.mul(gamma).add(beta);
    }

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

    // =========================================================================
    // DETAILED STEP-BY-STEP EXPLAINER LOGGERS (FORMULAE + INTUITION + MATRICES)
    // =========================================================================

    private static void printHeader() {
        System.out.println("==========================================================================");
        System.out.println("   COMPLETE TRANSFORMER ENCODER BLOCK: FORMULAE, INTUITION & MATRIX PIPELINE  ");
        System.out.println("==========================================================================");
    }

    private static void explainStep1(String[] tokens, NDArray E, float[][] rawEmbeddings, int dModel) {
        System.out.println("\n[STEP 1] SEMANTIC TOKEN EMBEDDING LOOKUP (E)");
        System.out.println("--------------------------------------------------------------------------");
        System.out.println("Intuition   : Converts discrete string tokens into continuous dense semantic vectors.");
        System.out.println("              Each word gets mapped to a coordinate point in a 384-dimensional space.");
        System.out.println("Formula     : E_i = Lookup(t_i) in R^{d_model}, where E in R^{N x d_model}");
        System.out.println("Tokens      : " + String.join(", ", tokens));
        System.out.println("Shape Out   : " + E.getShape() + " (N=" + tokens.length + ", d_model=" + dModel + ")\n");

        System.out.println("--- Embedding Matrix Slices E [N x d_model] (First 4 Dimensions Shown) ---");
        System.out.printf("%-6s | %-8s | %-45s%n", "Index", "Token", "Embedding Vector Slice E[i, 0..3]");
        System.out.println("-".repeat(68));

        for (int i = 0; i < tokens.length; i++) {
            float[] vec = rawEmbeddings[i];
            String snippet = String.format("[%.4f, %.4f, %.4f, %.4f, ...]", vec[0], vec[1], vec[2], vec[3]);
            System.out.printf("%-6d | %-8s | %-45s%n", i, tokens[i], snippet);
        }
        System.out.println("--------------------------------------------------------------------------");
    }

    private static void explainStep2(String[] tokens, NDArray P, NDArray X) {
        System.out.println("\n[STEP 2] POSITIONAL ENCODING & COMPOSITION (X = E + P)");
        System.out.println("--------------------------------------------------------------------------");
        System.out.println("Intuition   : Attention matrix ops are order-agnostic (permutation-equivariant).");
        System.out.println("              We inject sequence order coordinates using deterministic sine/cosine waves.");
        System.out.println("Formulas    : PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )");
        System.out.println("              PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )");
        System.out.println("              X = E + P in R^{N x d_model}");
        System.out.println("Shape Out   : " + X.getShape() + "\n");

        float[][] pArr = toFloatArray2D(P);
        float[][] xArr = toFloatArray2D(X);

        System.out.println("--- Matrix Values: Positional P vs. Composed Input X Matrix (First 4 Dims) ---");
        System.out.printf("%-4s | %-8s | %-28s | %-28s%n", "Pos", "Token", "Positional P[i, 0..3]", "Composed Input X=E+P [i, 0..3]");
        System.out.println("-".repeat(78));

        for (int i = 0; i < tokens.length; i++) {
            String pStr = String.format("[%.4f, %.4f, %.4f, %.4f]", pArr[i][0], pArr[i][1], pArr[i][2], pArr[i][3]);
            String xStr = String.format("[%.4f, %.4f, %.4f, %.4f]", xArr[i][0], xArr[i][1], xArr[i][2], xArr[i][3]);
            System.out.printf("%-4d | %-8s | %-28s | %-28s%n", i, tokens[i], pStr, xStr);
        }
        System.out.println("--------------------------------------------------------------------------");
    }

    private static void explainStep3(String[] tokens, int numHeads, int dModel, int dK,
                                     NDArray Q, NDArray K, NDArray V, NDArray Q_split,
                                     NDArray djlScores, NDArray attentionWeights, NDArray mhaOutput) {
        System.out.println("\n[STEP 3] SUBLAYER 1: MULTI-HEAD ATTENTION (MHA)");
        System.out.println("--------------------------------------------------------------------------");
        System.out.println("Intuition   : Instead of one global lookup, 8 parallel specialists (heads) evaluate");
        System.out.println("              syntax, semantics, and modifiers simultaneously in subspace d_k=48.");
        System.out.println("Formulas    : Q = X W_Q,  K = X W_K,  V = X W_V                  (Projections)");
        System.out.println("              Q_split, K_split, V_split in R^{h x N x d_k}             (Head Splitting)");
        System.out.println("              Scores = (Q_split * K_split^T) / sqrt(d_k)         (Scaled Inner Prod)");
        System.out.println("              A = Softmax(Scores, axis=-1)                       (Attention Weights)");
        System.out.println("              MHA(X) = Concat( A * V_split ) * W_O               (Concat & Project)");
        System.out.println("Shapes      : Q,K,V: " + Q.getShape() + " | Heads Split: " + Q_split.getShape() + 
                           " | Softmax A: " + attentionWeights.getShape() + " | MHA Out: " + mhaOutput.getShape() + "\n");

        float[][] qArr = toFloatArray2D(Q);
        float[][] kArr = toFloatArray2D(K);
        float[][] vArr = toFloatArray2D(V);

        System.out.println("--- Projections Matrix Slices: Q, K, V Matrices (First 3 Dims) ---");
        System.out.printf("%-8s | %-20s | %-20s | %-20s%n", "Token", "Query Q[i, 0..2]", "Key K[i, 0..2]", "Value V[i, 0..2]");
        System.out.println("-".repeat(74));

        for (int i = 0; i < tokens.length; i++) {
            String qStr = String.format("[%.4f, %.4f, %.4f]", qArr[i][0], qArr[i][1], qArr[i][2]);
            String kStr = String.format("[%.4f, %.4f, %.4f]", kArr[i][0], kArr[i][1], kArr[i][2]);
            String vStr = String.format("[%.4f, %.4f, %.4f]", vArr[i][0], vArr[i][1], vArr[i][2]);
            System.out.printf("%-8s | %-20s | %-20s | %-20s%n", tokens[i], qStr, kStr, vStr);
        }

        System.out.println("\n--- Head 0 Full 7x7 Softmax Attention Probability Matrix A_0 [N x N] ---");
        float[] flatWeights = attentionWeights.toFloatArray();
        int seqLen = tokens.length;

        System.out.printf("%-10s", "Q \\ K");
        for (String t : tokens) System.out.printf("%-9s", t);
        System.out.println("\n" + "-".repeat(10 + 9 * seqLen));

        for (int i = 0; i < seqLen; i++) {
            System.out.printf("%-10s", tokens[i]);
            for (int j = 0; j < seqLen; j++) {
                float w = flatWeights[i * seqLen + j]; // Head 0 offset starts at 0
                System.out.printf("%-9.4f", w);
            }
            System.out.println();
        }
        System.out.println("-".repeat(10 + 9 * seqLen));
        System.out.println("Note: Each row represents a Query token and sums to 1.0000 across all Key tokens.");
        System.out.println("--------------------------------------------------------------------------");
    }

    private static void explainStep4(String[] tokens, NDArray X, NDArray mhaOutput, 
                                     NDArray subLayer1Residual, NDArray subLayer1Output) {
        System.out.println("\n[STEP 4] SUBLAYER 1: RESIDUAL CONNECTION & LAYER NORM (ADD & NORM 1)");
        System.out.println("--------------------------------------------------------------------------");
        System.out.println("Intuition   : Residual connection acts as an information highway to preserve gradient flow.");
        System.out.println("              LayerNorm stabilizes activations across features (mean=0.0, var=1.0) per token.");
        System.out.println("Formulas    : SubLayer1_Residual = X + MHA(X)");
        System.out.println("              SubLayer1_Output   = LayerNorm(SubLayer1_Residual)");
        System.out.println("                                 = [(X_res - mean) / sqrt(var + eps)] * gamma + beta\n");

        System.out.println("--- SubLayer 1 Post-LayerNorm Verification Matrix Slices ---");
        System.out.printf("%-6s | %-8s | %-15s | %-15s | %-32s%n", "Index", "Token", "Post-Norm Mean", "Post-Norm Variance", "Normalized Slice [d0..d3]");
        System.out.println("-".repeat(88));

        NDArray normMeans = subLayer1Output.mean(new int[]{-1}, true);
        NDArray normVars = subLayer1Output.sub(normMeans).square().mean(new int[]{-1}, true);

        float[] meanFlat = normMeans.toFloatArray();
        float[] varFlat = normVars.toFloatArray();
        float[][] outArr = toFloatArray2D(subLayer1Output);

        for (int i = 0; i < tokens.length; i++) {
            String sliceStr = String.format("[%.4f, %.4f, %.4f, %.4f]", outArr[i][0], outArr[i][1], outArr[i][2], outArr[i][3]);
            System.out.printf("%-6d | %-8s | %-15.6f | %-15.6f | %-32s%n", i, tokens[i], meanFlat[i], varFlat[i], sliceStr);
        }
        System.out.println("--------------------------------------------------------------------------");
    }

    private static void explainStep5(String[] tokens, int dModel, int dFF, 
                                     NDArray linear1, NDArray ffnHidden, NDArray ffnOutput) {
        System.out.println("\n[STEP 5] SUBLAYER 2: POSITION-WISE FEED-FORWARD NETWORK (FFN)");
        System.out.println("--------------------------------------------------------------------------");
        System.out.println("Intuition   : Multi-Head Attention is a weighted linear combination of tokens.");
        System.out.println("              The FFN supplies non-linear capacity by projecting into a high-dimensional");
        System.out.println("              memory space (4x expansion to 1536) and back to store factual patterns.");
        System.out.println("Formulas    : Hidden  = ReLU( SubLayer1_Output * W1 + b1 )  -> Expands " + dModel + " to " + dFF);
        System.out.println("              FFN_Out = Hidden * W2 + b2                    -> Contracts " + dFF + " back to " + dModel);
        System.out.println("Shapes      : Hidden: " + ffnHidden.getShape() + " | FFN Output: " + ffnOutput.getShape() + "\n");

        float[][] hArr = toFloatArray2D(ffnHidden);
        float[][] oArr = toFloatArray2D(ffnOutput);

        System.out.println("--- FFN Matrix Slices: ReLU Hidden [N x 1536] vs Contracted FFN Output [N x 384] ---");
        System.out.printf("%-8s | %-32s | %-32s%n", "Token", "Expanded ReLU Hidden [d0..d3]", "Contracted FFN Out [d0..d3]");
        System.out.println("-".repeat(78));

        for (int i = 0; i < tokens.length; i++) {
            String hStr = String.format("[%.4f, %.4f, %.4f, %.4f]", hArr[i][0], hArr[i][1], hArr[i][2], hArr[i][3]);
            String oStr = String.format("[%.4f, %.4f, %.4f, %.4f]", oArr[i][0], oArr[i][1], oArr[i][2], oArr[i][3]);
            System.out.printf("%-8s | %-32s | %-32s%n", tokens[i], hStr, oStr);
        }
        System.out.println("--------------------------------------------------------------------------");
    }

    private static void explainStep6(String[] tokens, NDArray X, NDArray subLayer1, 
                                     NDArray ffnOutput, NDArray subLayer2Residual, NDArray finalEncoderOutput) {
        System.out.println("\n[STEP 6] SUBLAYER 2: RESIDUAL CONNECTION & LAYER NORM (ADD & NORM 2)");
        System.out.println("--------------------------------------------------------------------------");
        System.out.println("Intuition   : Adds the FFN output back to SubLayer1 output and normalizes feature vectors.");
        System.out.println("              Produces the final context-enriched vector representations for the Encoder.");
        System.out.println("Formulas    : SubLayer2_Residual = SubLayer1_Output + FFN(SubLayer1_Output)");
        System.out.println("              Encoder_Output     = LayerNorm(SubLayer2_Residual)");

        System.out.println("\n--- Complete Encoder Pipeline Tensor Shape Tracking ---");
        System.out.printf("%-40s | %-20s%n", "Encoder Pipeline Sub-Layer", "Tensor Shape");
        System.out.println("-".repeat(65));
        System.out.printf("%-40s | %-20s%n", "1. Input Composition (E + P)", X.getShape().toString());
        System.out.printf("%-40s | %-20s%n", "2. SubLayer 1 (MHA + Add & Norm)", subLayer1.getShape().toString());
        System.out.printf("%-40s | %-20s%n", "3. FFN Intermediate Expansion (Linear1)", ffnHiddenShape(tokens.length, X.getShape().get(1)));
        System.out.printf("%-40s | %-20s%n", "4. FFN Contraction Output (Linear2)", ffnOutput.getShape().toString());
        System.out.printf("%-40s | %-20s%n", "5. Final Encoder Block Output", finalEncoderOutput.getShape().toString());
        System.out.println("-".repeat(65));

        System.out.println("\n--- Final LayerNorm Statistical Verification & Matrix Values ---");
        System.out.printf("%-6s | %-8s | %-15s | %-15s | %-32s%n", "Index", "Token", "Post-Norm Mean", "Post-Norm Variance", "Final Vector Slice [d0..d3]");
        System.out.println("-".repeat(88));

        NDArray normMeans = finalEncoderOutput.mean(new int[]{-1}, true);
        NDArray normVars = finalEncoderOutput.sub(normMeans).square().mean(new int[]{-1}, true);

        float[] meanFlat = normMeans.toFloatArray();
        float[] varFlat = normVars.toFloatArray();
        float[][] outArr = toFloatArray2D(finalEncoderOutput);

        for (int i = 0; i < tokens.length; i++) {
            String sliceStr = String.format("[%.4f, %.4f, %.4f, %.4f]", outArr[i][0], outArr[i][1], outArr[i][2], outArr[i][3]);
            System.out.printf("%-6d | %-8s | %-15.6f | %-15.6f | %-32s%n", i, tokens[i], meanFlat[i], varFlat[i], sliceStr);
        }
        System.out.println("==========================================================================");
    }

    private static String ffnHiddenShape(long seqLen, long dModel) {
        return "[" + seqLen + ", " + (dModel * 4) + "]";
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