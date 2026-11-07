package com.example.math;

import ai.djl.inference.Predictor;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.GradientCollector;

import java.util.ArrayList;
import java.util.List;

public class ModernizedHierarchicalEncoderPipeline {

    public static void main(String[] args) throws Exception {
        printHeader();

        // Complex 28-token passage designed to demonstrate multi-layer semantic refinement
        String passage = "Although the financial bank situated near the river bank had closed its vault early due to rising floodwaters the regional director moved the remaining capital into reserve";
        String[] tokens = passage.split(" ");
        int seqLen = tokens.length;
        int numHeads = 8;
        int numLayers = 6; // Stack depth N = 6
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

            // Initialize Gradient Engine
            GradientEngine gradEngine = new GradientEngine(0.001f);
            List<NDArray> trainableParameters = new ArrayList<>();

            // =========================================================================
            // STAGE 1: SEMANTIC WORD EMBEDDINGS (E)
            // =========================================================================
            float[][] rawEmbeddings = new float[seqLen][];
            for (int i = 0; i < seqLen; i++) {
                rawEmbeddings[i] = predictor.predict(tokens[i]);
            }
            NDArray E = manager.create(rawEmbeddings);
            int dModel = (int) E.getShape().get(1); // d_model = 384
            int dFF = 4 * dModel;                   // d_ff = 1536

            explainStage1(tokens, E, rawEmbeddings, dModel);

            // =========================================================================
            // STAGE 2: ROTARY POSITION EMBEDDINGS (RoPE) PREPARATION & CAUSAL MASK
            // =========================================================================
            NDArray P = generatePositionalMatrix(manager, seqLen, dModel);
            NDArray currentX = E.duplicate();

            // Create 4D mask and squeeze to 2D [seqLen, seqLen] for broadcast multiplication
            NDArray causalMask4D = CausalMaskGenerator.createCausalMask(manager, seqLen);
            NDArray causalMask = causalMask4D.squeeze(0).squeeze(0);

            explainStage2(tokens, P, currentX, causalMask);

            // Polysemous token tracking: "financial bank" (idx 3) vs "river bank" (idx 8)
            int targetBank1 = 3;
            int targetBank2 = 8;
            float[][] bank1Trajectory = new float[numLayers + 1][4];
            float[][] bank2Trajectory = new float[numLayers + 1][4];

            saveTrajectorySlice(currentX, bank1Trajectory, 0, targetBank1);
            saveTrajectorySlice(currentX, bank2Trajectory, 0, targetBank2);

            System.out.println("==========================================================================");
            System.out.println("     BEGINNING SEQUENTIAL EXECUTION OF CAUSAL ENCODER PIPELINE (N = 6)    ");
            System.out.println("==========================================================================");

            // Build the stack ONCE: parameters persist across forward passes (required for training).
            int dK = dModel / numHeads;
            List<DecoderBlock> blocks = new ArrayList<>();
            for (int l = 0; l < numLayers; l++) {
                DecoderBlock block = new DecoderBlock(manager, dModel, numHeads, dFF, eps);
                blocks.add(block);
                trainableParameters.addAll(block.params());
            }
            NDArray[] ropeTables = buildRoPETables(manager, seqLen, dK); // shared by every layer

            // Open DJL Autograd Collector Tape Context
            try (GradientCollector collector = gradEngine.createCollector()) {

                // =========================================================================
                // STAGE 3: SEQUENTIAL 6-LAYER CAUSAL DECODER STACK EXECUTION
                // =========================================================================
                for (int l = 1; l <= numLayers; l++) {
                    System.out.println("\n" + "=".repeat(80));
                    System.out.println("              >>> CAUSAL TRANSFORMER LAYER " + l + " of " + numLayers + " <<<");
                    System.out.println("=".repeat(80));

                    DecoderBlock.Trace t = blocks.get(l - 1).forward(currentX, causalMask, ropeTables[0], ropeTables[1]);

                    explainLayerMHAStep(tokens, l, numHeads, dModel, dK, t.q(), t.k(), t.v(), t.qSplit(),
                            t.attention(), t.mhaOutput(), targetBank1);
                    explainLayerSubLayer1AddNorm(tokens, l, currentX, t.mhaOutput(), t.subLayer1Output(), t.subLayer1Output());
                    explainLayerFFNStep(tokens, l, dModel, dFF, t.ffnHidden(), t.ffnOutput());

                    double cosSimPrev = computeCosineSimilarity(currentX.get((long) targetBank1), t.output().get((long) targetBank1));
                    explainLayerSubLayer2AddNorm(tokens, l, currentX, t.subLayer1Output(), t.ffnOutput(), t.output(), cosSimPrev, targetBank1);

                    // Update input tensor state for the next layer in the stack
                    currentX = t.output();
                    saveTrajectorySlice(currentX, bank1Trajectory, l, targetBank1);
                    saveTrajectorySlice(currentX, bank2Trajectory, l, targetBank2);
                }
            }

            // =========================================================================
            // STAGE 4: DIVERGENCE AUDIT & HIERARCHICAL REFINEMENT SUMMARY
            // =========================================================================
            explainStackingTrajectorySummary(tokens, targetBank1, targetBank2, bank1Trajectory, bank2Trajectory);
        }
    }

    // =========================================================================
    // LAYER NORMALIZATION, SWISH, AND ROTARY POSITION EMBEDDING HELPERS
    // =========================================================================

    public static NDArray applySwish(NDArray X) {
        // Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
        NDArray denominator = X.neg().exp().add(1.0f);
        return X.div(denominator);
    }

    /**
     * Applies Rotary Position Embeddings (RoPE) using fully differentiable NDArray operations.
     * Preserves DJL autograd computation graph.
     * 
     * @param manager     DJL NDManager
     * @param splitTensor Input head tensor of shape [numHeads, seqLen, dK]
     * @return Rotated head tensor of shape [numHeads, seqLen, dK]
     */
    /**
     * Builds the RoPE cos/sin tables as [1, seqLen, dK] constants (each frequency repeated for its
     * (2i, 2i+1) pair). Build once per forward pass and reuse for every layer and for both Q and K.
     */
    public static NDArray[] buildRoPETables(NDManager manager, long seqLen, long dK) {
        // 1. Theta frequencies: theta_i = 10000^(-2i / dK)
        float[] invFreq = new float[(int) (dK / 2)];
        for (int i = 0; i < dK / 2; i++) {
            invFreq[i] = (float) (1.0 / Math.pow(10000.0, (2.0 * i) / dK));
        }
        NDArray invFreqND = manager.create(invFreq); // [dK / 2]

        // 2. Positions 0..seqLen-1, 3. outer product -> freqs [seqLen, dK / 2]
        NDArray posND = manager.arange(0, seqLen, 1, DataType.FLOAT32);
        NDArray freqs = posND.reshape(new Shape(seqLen, 1)).matMul(invFreqND.reshape(new Shape(1, dK / 2)));

        // 4. Duplicate each frequency for its paired dimensions: [seqLen, dK/2] -> [1, seqLen, dK]
        NDArray cos = freqs.cos().repeat(1, 2).reshape(new Shape(1, seqLen, dK));
        NDArray sin = freqs.sin().repeat(1, 2).reshape(new Shape(1, seqLen, dK));
        return new NDArray[]{cos, sin};
    }

    /** RoPE with precomputed tables. Pure NDArray ops, so autograd flows through it. */
    public static NDArray applyRoPE(NDArray splitTensor, NDArray cos, NDArray sin) {
        Shape shape = splitTensor.getShape();
        long numHeads = shape.get(0);
        long seqLen = shape.get(1);
        long dK = shape.get(2);

        // R(X) = [-x2, x1, -x4, x3, ...] via strided slices and interleaving stack
        NDArray x1 = splitTensor.get("..., 0::2"); // even features
        NDArray x2 = splitTensor.get("..., 1::2"); // odd features
        NDArray rx = x2.neg().stack(x1, -1).reshape(new Shape(numHeads, seqLen, dK));

        // RoPE formula: (X * cos) + (R(X) * sin)
        return splitTensor.mul(cos).add(rx.mul(sin));
    }

    /** Convenience wrapper: builds the tables on every call. Prefer the overload above in loops. */
    public static NDArray applyRoPE(NDManager manager, NDArray splitTensor) {
        Shape shape = splitTensor.getShape();
        NDArray[] tables = buildRoPETables(manager, shape.get(1), shape.get(2));
        return applyRoPE(splitTensor, tables[0], tables[1]);
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
    // STEP-BY-STEP AUDIT LOGGERS (FORMULAE + INTUITION + MATRIX SLICES)
    // =========================================================================

    private static void printHeader() {
        System.out.println("==========================================================================");
        System.out.println("  MODERNIZED TRANSFORMER ENCODER PIPELINE (Pre-LN + RoPE + SwiGLU / N=6) ");
        System.out.println("==========================================================================");
    }

    private static void explainStage1(String[] tokens, NDArray E, float[][] rawEmbeddings, int dModel) {
        System.out.println("\n[STAGE 1] SEMANTIC EMBEDDING LOOKUP (E)");
        System.out.println("--------------------------------------------------------------------------");
        System.out.println("Intuition   : Maps discrete string tokens into continuous dense semantic vectors.");
        System.out.println("Formula     : E_i = Lookup(t_i) in R^{d_model}, E in R^{N x d_model}");
        System.out.println("Passage     : \"" + String.join(" ", tokens) + "\"");
        System.out.println("Dimensions  : Sequence Length N = " + tokens.length + ", Model Dim d_model = " + dModel);
        System.out.println("Output Shape: " + E.getShape() + "\n");

        System.out.println("--- Embedding Slices E [N x 384] (First 6 Tokens, First 4 Dims) ---");
        System.out.printf("%-6s | %-12s | %-45s%n", "Index", "Token", "Embedding Vector Slice E[i, 0..3]");
        System.out.println("-".repeat(68));
        for (int i = 0; i < Math.min(6, tokens.length); i++) {
            float[] vec = rawEmbeddings[i];
            String snippet = String.format("[%.4f, %.4f, %.4f, %.4f, ...]", vec[0], vec[1], vec[2], vec[3]);
            System.out.printf("%-6d | %-12s | %-45s%n", i, tokens[i], snippet);
        }
        System.out.println("--------------------------------------------------------------------------");
    }

    private static void explainStage2(String[] tokens, NDArray P, NDArray X_0, NDArray causalMask) {
        System.out.println("\n[STAGE 2] ROTARY POSITION EMBEDDINGS (RoPE) PREPARATION & COMPOSITION (X_0 = E)");
        System.out.println("--------------------------------------------------------------------------");
        System.out.println("Intuition   : In Modern Pre-LN RoPE Transformer, positions are applied via 2D matrix rotations");
        System.out.println("              directly to Query and Key head tensors, leaving initial X_0 un-added.");
        System.out.println("Causal Mask : Upper-triangular matrix set to -1e9f to prevent future token leakage (j > i).");
        System.out.println("Formulas    : R_{theta, m} = [[cos(m*theta), -sin(m*theta)], [sin(m*theta), cos(m*theta)]]");
        System.out.println("              X_0 = E in R^{N x d_model}");
        System.out.println("Mask Shape  : " + causalMask.getShape());
        System.out.println("Output Shape: " + X_0.getShape() + "\n");

        float[][] pArr = toFloatArray2D(P);
        float[][] xArr = toFloatArray2D(X_0);

        System.out.println("--- Reference Frequency P Matrix vs. Unadulterated Initial X_0 (First 6 Tokens, 4 Dims) ---");
        System.out.printf("%-4s | %-12s | %-28s | %-28s%n", "Pos", "Token", "Reference P[i, 0..3]", "Initial Input X_0 = E [i, 0..3]");
        System.out.println("-".repeat(80));
        for (int i = 0; i < Math.min(6, tokens.length); i++) {
            String pStr = String.format("[%.4f, %.4f, %.4f, %.4f]", pArr[i][0], pArr[i][1], pArr[i][2], pArr[i][3]);
            String xStr = String.format("[%.4f, %.4f, %.4f, %.4f]", xArr[i][0], xArr[i][1], xArr[i][2], xArr[i][3]);
            System.out.printf("%-4d | %-12s | %-28s | %-28s%n", i, tokens[i], pStr, xStr);
        }
        System.out.println("--------------------------------------------------------------------------");
    }

    private static void explainLayerMHAStep(String[] tokens, int layerIdx, int numHeads, int dModel, int dK,
                                            NDArray Q, NDArray K, NDArray V, NDArray Q_split,
                                            NDArray attentionWeights, NDArray mhaOutput, int targetIdx) {
        System.out.println("\n[LAYER " + layerIdx + " - STEP 1] Multi-Head Attention Sub-layer (Pre-LN + RoPE + Causal Mask)");
        System.out.println("--------------------------------------------------------------------------");
        System.out.println("Intuition   : Pre-LN normalizes feature inputs before projection. RoPE rotates Q and K.");
        System.out.println("Formulas    : LN_In = LayerNorm(X_{l-1}), Q = LN_In * W_Q, K = LN_In * W_K, V = LN_In * W_V");
        System.out.println("              Q_rope = ApplyRoPE(Q_split), K_rope = ApplyRoPE(K_split)");
        System.out.println("              Scores = (Q_rope * K_rope^T) / sqrt(d_k) + CausalMask,  A = Softmax(Scores, axis=-1)");
        System.out.println("Shapes      : Q,K,V: " + Q.getShape() + " | Heads Split: " + Q_split.getShape() + 
                           " | Softmax A: " + attentionWeights.getShape() + " | MHA Out: " + mhaOutput.getShape() + "\n");

        float[][] qArr = toFloatArray2D(Q);
        float[][] kArr = toFloatArray2D(K);
        float[][] vArr = toFloatArray2D(V);

        System.out.println("--- Projection Slices for Token '" + tokens[targetIdx] + "' (Index " + targetIdx + ") ---");
        String qStr = String.format("[%.4f, %.4f, %.4f, %.4f]", qArr[targetIdx][0], qArr[targetIdx][1], qArr[targetIdx][2], qArr[targetIdx][3]);
        String kStr = String.format("[%.4f, %.4f, %.4f, %.4f]", kArr[targetIdx][0], kArr[targetIdx][1], kArr[targetIdx][2], kArr[targetIdx][3]);
        String vStr = String.format("[%.4f, %.4f, %.4f, %.4f]", vArr[targetIdx][0], vArr[targetIdx][1], vArr[targetIdx][2], vArr[targetIdx][3]);
        System.out.println("  -> Query Q[" + targetIdx + ", 0..3] : " + qStr);
        System.out.println("  -> Key   K[" + targetIdx + ", 0..3] : " + kStr);
        System.out.println("  -> Value V[" + targetIdx + ", 0..3] : " + vStr);

        System.out.println("\n--- Head 0 Causal Attention Focus for '" + tokens[targetIdx] + "' (Index " + targetIdx + ") ---");
        float[] flatWeights = attentionWeights.toFloatArray();
        int seqLen = tokens.length;
        printHeadTopWeight(tokens, flatWeights, 0, seqLen, targetIdx);
        System.out.println("--------------------------------------------------------------------------");
    }

    private static void explainLayerSubLayer1AddNorm(String[] tokens, int layerIdx, NDArray X_prev, NDArray mhaOut,
                                                     NDArray subLayer1Residual, NDArray subLayer1Output) {
        System.out.println("\n[LAYER " + layerIdx + " - STEP 2] SubLayer 1 Pre-LN Residual Highway Addition");
        System.out.println("--------------------------------------------------------------------------");
        System.out.println("Intuition   : Pre-LN adds MHA delta updates directly to main stream without normalizing output.");
        System.out.println("Formulas    : Sub1_Out = X_{l-1} + CausalMHA( LayerNorm(X_{l-1}) )");

        NDArray normMeans = subLayer1Output.mean(new int[]{-1}, true);
        NDArray normVars = subLayer1Output.sub(normMeans).square().mean(new int[]{-1}, true);

        float[] meanFlat = normMeans.toFloatArray();
        float[] varFlat = normVars.toFloatArray();

        System.out.printf("Residual Stream Check : Layer %d SubLayer 1 Stream Mean = %.6f | Stream Variance = %.6f%n", 
                layerIdx, meanFlat[0], varFlat[0]);
        System.out.println("--------------------------------------------------------------------------");
    }

    private static void explainLayerFFNStep(String[] tokens, int layerIdx, int dModel, int dFF,
                                            NDArray ffnHidden, NDArray ffnOutput) {
        System.out.println("\n[LAYER " + layerIdx + " - STEP 3] Position-wise SwiGLU Feed-Forward Network");
        System.out.println("--------------------------------------------------------------------------");
        System.out.println("Intuition   : SwiGLU utilizes parallel gate and feature projections with Swish activation.");
        System.out.println("Formulas    : Gate    = Swish( LN_In * W_g^{(l)} + b_g^{(l)} )");
        System.out.println("              Feature = LN_In * W_1^{(l)} + b_1^{(l)}");
        System.out.println("              FFN_Out = (Gate (x) Feature) * W_2^{(l)} + b_2^{(l)}");
        System.out.println("Shapes Out  : SwiGLU Hidden Activation: " + ffnHidden.getShape() + " | FFN Output: " + ffnOutput.getShape());
        System.out.println("--------------------------------------------------------------------------");
    }

    private static void explainLayerSubLayer2AddNorm(String[] tokens, int layerIdx, NDArray X_prev, NDArray subLayer1Out,
                                                     NDArray ffnOutput, NDArray layerOutput, double cosSimPrev, int targetIdx) {
        System.out.println("\n[LAYER " + layerIdx + " - STEP 4] SubLayer 2 Pre-LN Residual Stream Addition");
        System.out.println("--------------------------------------------------------------------------");
        System.out.println("Intuition   : Merges non-linear SwiGLU delta update into main residual highway.");
        System.out.println("Formulas    : X_l = Sub1_Out + SwiGLU( LayerNorm(Sub1_Out) )");

        NDArray normMeans = layerOutput.mean(new int[]{-1}, true);
        NDArray normVars = layerOutput.sub(normMeans).square().mean(new int[]{-1}, true);

        float[] meanFlat = normMeans.toFloatArray();
        float[] varFlat = normVars.toFloatArray();
        float[][] outArr = toFloatArray2D(layerOutput);

        String sliceStr = String.format("[%.4f, %.4f, %.4f, %.4f]", outArr[targetIdx][0], outArr[targetIdx][1], outArr[targetIdx][2], outArr[targetIdx][3]);

        System.out.printf("Layer %d Output Shape: %s%n", layerIdx, layerOutput.getShape());
        System.out.printf("Layer %d Stream Check: Mean = %.6f | Variance = %.6f%n", layerIdx, meanFlat[targetIdx], varFlat[targetIdx]);
        System.out.printf("Token '%s'[%d] Slice : %s%n", tokens[targetIdx], targetIdx, sliceStr);
        System.out.printf("Contextual Shift    : Cosine Similarity to Layer %d = %.4f%n", (layerIdx - 1), cosSimPrev);
        System.out.println("--------------------------------------------------------------------------");
    }

    private static void explainStackingTrajectorySummary(String[] tokens, int idx1, int idx2,
                                                          float[][] traj1, float[][] traj2) {
        System.out.println("\n" + "=".repeat(88));
        System.out.println("   MODERNIZED CAUSAL POLYSEMY DISAMBIGUATION & STACKING TRAJECTORY SUMMARY (V6)   ");
        System.out.println("=".repeat(88));
        System.out.println("Passage Excerpt: '...financial bank [idx 3] ... river bank [idx 8]...'");
        System.out.println("Core Objective : Evidence how un-adulterated embeddings (X_0) refine across Pre-LN/RoPE/SwiGLU layers.");
        System.out.println("-".repeat(88));

        String[] layerStages = {
            "Input (X_0)", "Layer 1 (X_1)", "Layer 2 (X_2)", 
            "Layer 3 (X_3)", "Layer 4 (X_4)", "Layer 5 (X_5)", "Layer 6 (X_6)"
        };

        String[] descriptions = {
            "Static Lexical Embedding (Un-adulterated E)",
            "Pre-LN RoPE Causal Grammatical & Part-of-Speech Association",
            "Mid-range Causal Clause Disambiguation via RoPE Relative Rotation",
            "Coreference Linkage via SwiGLU Gated Activation ('financial bank' -> 'its vault')",
            "Multi-Hop Causal Reasoning ('floodwaters' -> 'reserve')",
            "Global Sentence Synthesis & Pre-LN Causal Stream Alignment",
            "Final Fully-Enriched Deep Contextual Representation"
        };

        System.out.printf("%-13s | %-30s | %-30s%n", "Layer Stage", "Financial 'bank' [idx 3] Slice", "River 'bank' [idx 8] Slice");
        System.out.println("-".repeat(88));

        for (int l = 0; l <= 6; l++) {
            String slice1 = String.format("[%.4f, %.4f, %.4f, %.4f]", traj1[l][0], traj1[l][1], traj1[l][2], traj1[l][3]);
            String slice2 = String.format("[%.4f, %.4f, %.4f, %.4f]", traj2[l][0], traj2[l][1], traj2[l][2], traj2[l][3]);
            System.out.printf("%-13s | %-30s | %-30s%n", layerStages[l], slice1, slice2);
        }
        System.out.println("-".repeat(88));

        System.out.println("\n--- Modernized Layer-by-Layer Functional Role Breakdown ---");
        for (int l = 0; l <= 6; l++) {
            System.out.printf("  -> %-13s : %s%n", layerStages[l], descriptions[l]);
        }
        System.out.println("==========================================================================");
    }

    private static void printHeadTopWeight(String[] tokens, float[] flatWeights, int headIdx, int seqLen, int queryRow) {
        int headOffset = headIdx * seqLen * seqLen;
        int rowOffset = headOffset + queryRow * seqLen;

        int maxCol = 0;
        float maxVal = -1.0f;

        // Restrict search bounds to j <= queryRow to honor Causal Masking
        for (int j = 0; j <= queryRow; j++) {
            float val = flatWeights[rowOffset + j];
            if (val > maxVal) {
                maxVal = val;
                maxCol = j;
            }
        }
        System.out.printf("  -> Head %d: '%s' (idx %d) attends causally to '%s' (idx %d) [Weight = %.4f]%n",
                headIdx, tokens[queryRow], queryRow, tokens[maxCol], maxCol, maxVal);
    }

    private static double computeCosineSimilarity(NDArray vecA, NDArray vecB) {
        float[] a = vecA.toFloatArray();
        float[] b = vecB.toFloatArray();

        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    private static void saveTrajectorySlice(NDArray matrix, float[][] trajectory, int layer, int tokenIdx) {
        float[] row = matrix.get((long) tokenIdx).toFloatArray();
        System.arraycopy(row, 0, trajectory[layer], 0, 4);
    }

    private static float[][][] to3DArray(NDArray array, int d0, int d1, int d2) {
        float[] flat = array.toFloatArray();
        float[][][] result = new float[d0][d1][d2];
        int idx = 0;
        for (int i = 0; i < d0; i++) {
            for (int j = 0; j < d1; j++) {
                for (int k = 0; k < d2; k++) {
                    result[i][j][k] = flat[idx++];
                }
            }
        }
        return result;
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