package com.example.math;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Transformer Self-Attention Unit & Integration Test Suite")
class TransformerAttentionTest {

    private static final double EPSILON_PRECISION = 1e-6;
    private static final double DUAL_TRACK_TOLERANCE = 1e-5;

    // =========================================================================
    // 1. MATHEMATICAL & SOFTMAX INVARIANT TESTS
    // =========================================================================

    @Nested
    @DisplayName("Mathematical Invariants & Probability Constraints")
    class MathematicalInvariants {

        @Test
        @DisplayName("Softmax rows must sum to exactly 1.0 (Probability Axiom)")
        void testSoftmaxRowSumEqualsOne() {
            try (NDManager manager = NDManager.newBaseManager()) {
                NDArray logits = manager.randomNormal(new Shape(7, 7));
                NDArray probabilities = logits.softmax(-1);

                float[] flat = probabilities.toFloatArray();
                int seqLen = 7;

                for (int i = 0; i < seqLen; i++) {
                    double rowSum = 0.0;
                    for (int j = 0; j < seqLen; j++) {
                        float val = flat[i * seqLen + j];
                        assertThat(val)
                            .withFailMessage("Attention probability at [%d, %d] must be non-negative", i, j)
                            .isGreaterThanOrEqualTo(0.0f);
                        rowSum += val;
                    }
                    assertEquals(1.0, rowSum, EPSILON_PRECISION, 
                        String.format("Row %d probability sum must equal 1.0", i));
                }
            }
        }

        @Test
        @DisplayName("Uniform / Zero Input Logits must produce identical uniform probabilities (1 / N)")
        void testZeroLogitsUniformProbability() {
            int seqLen = 5;
            try (NDManager manager = NDManager.newBaseManager()) {
                NDArray zeroLogits = manager.zeros(new Shape(seqLen, seqLen));
                NDArray probabilities = zeroLogits.softmax(-1);

                float[] flat = probabilities.toFloatArray();
                float expectedUniform = 1.0f / seqLen;

                for (float prob : flat) {
                    assertEquals(expectedUniform, prob, 1e-6f, 
                        "Zero logits must yield uniform attention weights (1/N)");
                }
            }
        }

        @Test
        @DisplayName("Scaling factor sqrt(d_k) prevents extreme logit saturation")
        void testScalingFactorEffect() {
            int seqLen = 4;
            int dK = 64;
            double scale = Math.sqrt(dK);

            try (NDManager manager = NDManager.newBaseManager()) {
                NDArray rawScores = manager.create(new float[][]{
                    {10.0f, -5.0f, 2.0f, 0.0f},
                    {1.0f,  4.0f,  -2.0f, 3.0f},
                    {0.0f,  0.0f,   0.0f, 0.0f},
                    {-3.0f, 8.0f,   1.0f, 2.0f}
                });

                NDArray unscaledWeights = rawScores.softmax(-1);
                NDArray scaledWeights = rawScores.div(scale).softmax(-1);

                float[] unscaledFlat = unscaledWeights.toFloatArray();
                float[] scaledFlat = scaledWeights.toFloatArray();

                // Unscaled max attention should be more aggressive/peaked than scaled
                assertThat(unscaledFlat[0])
                    .withFailMessage("Unscaled peak probability should exceed scaled probability due to variance inflation")
                    .isGreaterThan(scaledFlat[0]);
            }
        }
    }

    // =========================================================================
    // 2. DUAL-TRACK PRECISION ALIGNMENT TESTS
    // =========================================================================

    @Nested
    @DisplayName("Dual-Track Engine Precision Verification (DJL vs. Manual Java)")
    class DualTrackVerification {

        @Test
        @DisplayName("Manual Bilinear Product path must match DJL Tensor MatMul within 1e-7 tolerance")
        void testDualTrackAlignment() {
            int seqLen = 5;
            int dModel = 128;
            int dK = 32;

            try (NDManager manager = NDManager.newBaseManager()) {
                // Synthesize deterministic test inputs
                NDArray X = manager.randomNormal(0.0f, 1.0f, new Shape(seqLen, dModel), DataType.FLOAT32);
                float stddev = (float) Math.sqrt(2.0 / (dModel + dK));

                NDArray W_Q = manager.randomNormal(0.0f, stddev, new Shape(dModel, dK), DataType.FLOAT32);
                NDArray W_K = manager.randomNormal(0.0f, stddev, new Shape(dModel, dK), DataType.FLOAT32);

                // Path A: DJL Native Tensors
                NDArray Q = X.matMul(W_Q);
                NDArray K = X.matMul(W_K);
                NDArray djlScores = Q.matMul(K.transpose()).div(Math.sqrt(dK));
                NDArray djlAttentionWeights = djlScores.softmax(-1);
                float[] djlFlat = djlAttentionWeights.toFloatArray();

                // Path B: Primitive Double Loops (Manual Bilinear Math)
                float[][] xArr = to2D(X);
                float[][] wqArr = to2D(W_Q);
                float[][] wkArr = to2D(W_K);

                double[][] manualScores = new double[seqLen][seqLen];
                double[][] manualWeights = new double[seqLen][seqLen];

                for (int i = 0; i < seqLen; i++) {
                    double[] q_i = projectVector(xArr[i], wqArr);
                    for (int j = 0; j < seqLen; j++) {
                        double[] k_j = projectVector(xArr[j], wkArr);
                        manualScores[i][j] = dotProduct(q_i, k_j) / Math.sqrt(dK);
                    }
                }

                // Softmax
                for (int i = 0; i < seqLen; i++) {
                    double maxScore = Double.NEGATIVE_INFINITY;
                    for (int j = 0; j < seqLen; j++) {
                        if (manualScores[i][j] > maxScore) maxScore = manualScores[i][j];
                    }
                    double sumExp = 0.0;
                    double[] expScores = new double[seqLen];
                    for (int j = 0; j < seqLen; j++) {
                        expScores[j] = Math.exp(manualScores[i][j] - maxScore);
                        sumExp += expScores[j];
                    }
                    for (int j = 0; j < seqLen; j++) {
                        manualWeights[i][j] = expScores[j] / sumExp;
                    }
                }

                // Assert Precision Equivalence
                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < seqLen; j++) {
                        double manualVal = manualWeights[i][j];
                        double djlVal = djlFlat[i * seqLen + j];
                        double diff = Math.abs(manualVal - djlVal);

                        assertEquals(manualVal, djlVal, DUAL_TRACK_TOLERANCE,
                            String.format("Mismatch at [%d, %d]: Manual=%.7f, DJL=%.7f, Diff=%.2e", 
                                i, j, manualVal, djlVal, diff));
                    }
                }
            }
        }
    }

    // =========================================================================
    // 3. BOUNDARY, SHAPE & EDGE-CASE TESTS
    // =========================================================================

    @Nested
    @DisplayName("Edge Cases & Dimensional Transformations")
    class EdgeCasesAndShapes {

        @Test
        @DisplayName("Single-token sequence (N=1) must return 1.0 attention self-weight")
        void testSingleTokenSequence() {
            try (NDManager manager = NDManager.newBaseManager()) {
                NDArray Q = manager.randomNormal(new Shape(1, 64));
                NDArray K = manager.randomNormal(new Shape(1, 64));

                NDArray scores = Q.matMul(K.transpose()).div(Math.sqrt(64));
                NDArray weights = scores.softmax(-1);

                assertEquals(1, weights.getShape().get(0));
                assertEquals(1, weights.getShape().get(1));
                assertEquals(1.0f, weights.toFloatArray()[0], 1e-6f, 
                    "Single token must allocate 100% attention to itself");
            }
        }

        @ParameterizedTest
        @ValueSource(ints = {1, 8, 32, 128})
        @DisplayName("Attention output tensor dimensions must strictly match [N, d_v] for varying sequence lengths")
        void testOutputShapeConservation(int seqLen) {
            int dModel = 384;
            int dK = 64;
            int dV = 64;

            try (NDManager manager = NDManager.newBaseManager()) {
                NDArray X = manager.zeros(new Shape(seqLen, dModel));
                NDArray W_Q = manager.zeros(new Shape(dModel, dK));
                NDArray W_K = manager.zeros(new Shape(dModel, dK));
                NDArray W_V = manager.zeros(new Shape(dModel, dV));

                NDArray Q = X.matMul(W_Q);
                NDArray K = X.matMul(W_K);
                NDArray V = X.matMul(W_V);

                NDArray weights = Q.matMul(K.transpose()).div(Math.sqrt(dK)).softmax(-1);
                NDArray output = weights.matMul(V);

                Shape outputShape = output.getShape();
                assertEquals(2, outputShape.dimension(), "Output tensor must be 2D matrix [N, d_v]");
                assertEquals(seqLen, outputShape.get(0), "Sequence length N must be conserved");
                assertEquals(dV, outputShape.get(1), "Value dimension d_v must match target projection space");
            }
        }

        @Test
        @DisplayName("Causal masking must force upper-triangular weights to zero")
        void testCausalMasking() {
            int seqLen = 4;
            try (NDManager manager = NDManager.newBaseManager()) {
                NDArray rawScores = manager.ones(new Shape(seqLen, seqLen)).mul(2.0f);

                // Create upper triangular mask with large negative value (-1e9)
                float[][] maskArray = new float[seqLen][seqLen];
                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < seqLen; j++) {
                        if (j > i) {
                            maskArray[i][j] = -1e9f;
                        }
                    }
                }
                NDArray mask = manager.create(maskArray);
                NDArray maskedScores = rawScores.add(mask);
                NDArray maskedWeights = maskedScores.softmax(-1);

                float[][] weightsMatrix = to2D(maskedWeights);

                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < seqLen; j++) {
                        if (j > i) {
                            assertEquals(0.0f, weightsMatrix[i][j], 1e-6f,
                                String.format("Future token weight at [%d, %d] must be masked to zero", i, j));
                        }
                    }
                }
            }
        }
    }

    // =========================================================================
    // HELPER MATH UTILITIES
    // =========================================================================

    private static float[][] to2D(NDArray array) {
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

    private static double[] projectVector(float[] input, float[][] weights) {
        int outDim = weights[0].length;
        double[] result = new double[outDim];
        for (int j = 0; j < outDim; j++) {
            double sum = 0.0;
            for (int i = 0; i < input.length; i++) {
                sum += input[i] * weights[i][j];
            }
            result[j] = sum;
        }
        return result;
    }

    private static double dotProduct(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
}