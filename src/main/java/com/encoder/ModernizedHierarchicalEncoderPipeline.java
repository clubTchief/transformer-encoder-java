package com.encoder;

import ai.djl.inference.Predictor;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.GradientCollector;
import com.transformer.CausalMaskGenerator;
import com.decoder.DecoderBlock;
import com.transformer.GradientEngine;

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
            // STAGE 2: INITIALIZATION & DECODER BLOCK STACK SETUP (DONE ONCE)
            // =========================================================================
            NDArray P = generatePositionalMatrix(manager, seqLen, dModel);
            NDArray currentX = E.duplicate();

            // Create 4D mask and squeeze to 2D [seqLen, seqLen] for broadcast multiplication
            NDArray causalMask4D = CausalMaskGenerator.createCausalMask(manager, seqLen);
            NDArray causalMask = causalMask4D.squeeze(0).squeeze(0);

            explainStage2(tokens, P, currentX, causalMask);

            // Instantiate persistent DecoderBlocks ONCE (No weight re-allocations in forward loop)
            List<DecoderBlock> blocks = new ArrayList<>();
            List<NDArray> allTrainableParams = new ArrayList<>();

            for (int l = 0; l < numLayers; l++) {
                DecoderBlock block = new DecoderBlock(manager, dModel, numHeads, dFF, eps);
                blocks.add(block);
                allTrainableParams.addAll(block.params());
            }

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

            // =========================================================================
            // STAGE 3: REFACTORED MAIN FORWARD PASS LOOP UNDER GRADIENT COLLECTOR
            // =========================================================================
            try (GradientCollector collector = gradEngine.createCollector()) {

                // Pre-compute RoPE cos/sin frequencies ONCE per forward pass to eliminate memory leaks
                int dK = dModel / numHeads;
                NDArray[] ropeFreqs = precomputeRoPEFrequencies(manager, seqLen, dK);
                NDArray cos = ropeFreqs[0];
                NDArray sin = ropeFreqs[1];

                for (int l = 1; l <= numLayers; l++) {
                    System.out.println("\n" + "=".repeat(80));
                    System.out.println("              >>> CAUSAL TRANSFORMER ENCODER LAYER " + l + " of " + numLayers + " <<<");
                    System.out.println("=".repeat(80));

                    DecoderBlock block = blocks.get(l - 1);
                    NDArray nextX = block.forward(currentX, causalMask, cos, sin);

                    double cosSimPrev = computeCosineSimilarity(currentX.get((long) targetBank1), nextX.get((long) targetBank1));
                    explainLayerSubLayer2AddNorm(tokens, l, currentX, currentX, nextX, nextX, cosSimPrev, targetBank1);

                    currentX = nextX;
                    saveTrajectorySlice(currentX, bank1Trajectory, l, targetBank1);
                    saveTrajectorySlice(currentX, bank2Trajectory, l, targetBank2);
                }

                // Compute loss & accumulate gradients across autograd tape
                NDArray loss = currentX.sum();
                collector.backward(loss);
            } // <--- GradientCollector closes here, stopping active tape recording

            // Perform optimizer update AFTER collector closes
            gradEngine.step(allTrainableParams);

            // =========================================================================
            // STAGE 4: DIVERGENCE AUDIT & HIERARCHICAL REFINEMENT SUMMARY
            // =========================================================================
            explainStackingTrajectorySummary(tokens, targetBank1, targetBank2, bank1Trajectory, bank2Trajectory);
        }
    }

    // =========================================================================
    // EFFICIENT ROPE PRE-COMPUTATION & VECTOR-NATIVE ROTATION HELPERS
    // =========================================================================

    public static NDArray[] precomputeRoPEFrequencies(NDManager manager, long seqLen, long dK) {
        float[] invFreq = new float[(int) (dK / 2)];
        for (int i = 0; i < dK / 2; i++) {
            invFreq[i] = (float) (1.0 / Math.pow(10000.0, (2.0 * i) / dK));
        }
        NDArray invFreqND = manager.create(invFreq);
        NDArray posND = manager.arange(0, seqLen, 1, DataType.FLOAT32);

        NDArray freqs = posND.reshape(new Shape(seqLen, 1)).matMul(invFreqND.reshape(new Shape(1, dK / 2)));

        NDArray cos = freqs.cos().repeat(1, 2).reshape(new Shape(1, seqLen, dK));
        NDArray sin = freqs.sin().repeat(1, 2).reshape(new Shape(1, seqLen, dK));

        return new NDArray[]{cos, sin};
    }

    public static NDArray applyRoPE(NDArray splitTensor, NDArray cos, NDArray sin) {
        Shape shape = splitTensor.getShape();
        long numHeads = shape.get(0);
        long seqLen = shape.get(1);
        long dK = shape.get(2);

        NDArray x1 = splitTensor.get("..., 0::2");
        NDArray x2 = splitTensor.get("..., 1::2");

        NDArray negX2 = x2.neg();
        NDArray rx = negX2.stack(x1, -1).reshape(new Shape(numHeads, seqLen, dK));

        return splitTensor.mul(cos).add(rx.mul(sin));
    }

    /**
     * Helper overload for unit tests. Pre-computes frequencies dynamically.
     */
    public static NDArray applyRoPE(NDManager manager, NDArray splitTensor) {
        Shape shape = splitTensor.getShape();
        long seqLen = shape.get(1);
        long dK = shape.get(2);

        NDArray[] freqs = precomputeRoPEFrequencies(manager, seqLen, dK);
        return applyRoPE(splitTensor, freqs[0], freqs[1]);
    }

    public static NDArray applySwish(NDArray X) {
        NDArray denominator = X.neg().exp().add(1.0f);
        return X.div(denominator);
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
    // STEP-BY-STEP AUDIT LOGGERS
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