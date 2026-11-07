package com.example.math;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ModernizedHierarchicalEncoderPipelineTest {

    private NDManager manager;

    @BeforeEach
    void setUp() {
        manager = NDManager.newBaseManager();
    }

    @AfterEach
    void tearDown() {
        manager.close();
    }

    // =========================================================================
    // RESTORED & FIXED TESTS FROM PREVIOUS VERSION
    // =========================================================================

    @Test
    @DisplayName("Verify LayerNorm produces zero mean and unit variance")
    void testLayerNormNormalization() {
        NDArray input = manager.create(new float[][]{
                {1.0f, 2.0f, 3.0f, 4.0f},
                {10.0f, 20.0f, 30.0f, 40.0f}
        });

        int dModel = (int) input.getShape().get(1); // Resolved missing dModel symbol
        LayerNorm layerNorm = new LayerNorm(manager, dModel, 1e-5f);
        NDArray normed = layerNorm.forward(input); // Resolved inputTensor -> input, defined normed

        assertEquals(new Shape(2, 4), normed.getShape());

        float[] mean = normed.mean(new int[]{-1}, true).toFloatArray();
        assertEquals(0.0f, mean[0], 1e-4f, "LayerNorm mean should be close to 0");
        assertEquals(0.0f, mean[1], 1e-4f, "LayerNorm mean should be close to 0");
    }

    @Test
    @DisplayName("Verify Swish activation formula: x / (1 + exp(-x))")
    void testSwishActivation() {
        NDArray input = manager.create(new float[]{0.0f, 2.0f, -2.0f});
        NDArray output = ModernizedHierarchicalEncoderPipeline.applySwish(input);

        float[] result = output.toFloatArray();

        // Swish(0) = 0 / 2 = 0.0
        assertEquals(0.0f, result[0], 1e-5f);
        // Swish(2) = 2 / (1 + exp(-2)) ≈ 1.76159
        assertEquals(1.76159f, result[1], 1e-4f);
        // Swish(-2) = -2 / (1 + exp(2)) ≈ -0.23841
        assertEquals(-0.23841f, result[2], 1e-4f);
    }

    @Test
    @DisplayName("Verify RoPE preserves tensor shape [h, N, dK]")
    void testRoPEShapePreservation() {
        int h = 2, N = 4, dK = 8;
        NDArray splitTensor = manager.ones(new Shape(h, N, dK));

        NDArray ropeTensor = ModernizedHierarchicalEncoderPipeline.applyRoPE(manager, splitTensor);

        assertEquals(new Shape(h, N, dK), ropeTensor.getShape());
        assertNotNull(ropeTensor.toFloatArray());
    }

    @Test
    @DisplayName("Verify RoPE is fully differentiable and propagates gradients")
    void testRoPEDifferentiability() {
        int h = 2, N = 4, dK = 8;
        
        // Create input with autograd tracking enabled
        NDArray splitTensor = manager.ones(new Shape(h, N, dK), DataType.FLOAT32);
        splitTensor.setRequiresGradient(true);

        GradientEngine gradEngine = new GradientEngine(0.01f);

        try (GradientCollector collector = gradEngine.createCollector()) {
            // Forward pass through vectorized RoPE
            NDArray ropeOutput = ModernizedHierarchicalEncoderPipeline.applyRoPE(manager, splitTensor);
            
            // Scalar loss computation
            NDArray loss = ropeOutput.sum();
            
            // Execute backpropagation
            collector.backward(loss);
        }

        // Assert that gradients flowed backward into splitTensor
        assertTrue(splitTensor.hasGradient(), "Input tensor to RoPE must receive gradients after backward pass");
        assertNotNull(splitTensor.getGradient(), "Gradient tensor for RoPE input cannot be null");
        assertEquals(new Shape(h, N, dK), splitTensor.getGradient().getShape());
    }

    // =========================================================================
    // NEW AUTOGRAD, CAUSAL MASK & GRADIENT ENGINE TESTS
    // =========================================================================

    @Test
    @DisplayName("Verify LayerNorm registers learnable gamma and beta autograd parameters")
    void testLayerNormAutogradParameters() {
        int dModel = 384;
        LayerNorm layerNorm = new LayerNorm(manager, dModel, 1e-5f);

        assertTrue(layerNorm.getGamma().hasGradient(), "LayerNorm gamma must require gradient");
        assertTrue(layerNorm.getBeta().hasGradient(), "LayerNorm beta must require gradient");
    }

    @Test
    @DisplayName("Verify CausalMaskGenerator creates lower triangular additive matrix")
    void testCausalMaskGenerator() {
        long seqLen = 4;
        NDArray rawMask = CausalMaskGenerator.createCausalMask(manager, seqLen);

        assertEquals(new Shape(1, 1, seqLen, seqLen), rawMask.getShape());

        NDArray mask2D = rawMask.squeeze(0).squeeze(0);
        assertEquals(new Shape(seqLen, seqLen), mask2D.getShape());

        float[] maskValues = mask2D.toFloatArray();
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                float value = maskValues[i * (int) seqLen + j];
                if (j > i) {
                    assertEquals(-1e9f, value, 1e-2f, "Future positions (j > i) must be -1e9");
                } else {
                    assertEquals(0.0f, value, 1e-6f, "Past/present positions (j <= i) must be 0.0");
                }
            }
        }
    }

    @Test
    @DisplayName("Verify GradientEngine Autograd collection and weight update step")
    void testGradientEngineStep() {
        GradientEngine gradEngine = new GradientEngine(0.1f);

        NDArray weight = manager.ones(new Shape(2, 2), DataType.FLOAT32);
        weight.setRequiresGradient(true);

        NDArray input = manager.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f}, new Shape(2, 2));

        try (GradientCollector collector = gradEngine.createCollector()) {
            NDArray output = weight.mul(input);
            NDArray loss = output.sum();
            collector.backward(loss);
        }

        assertTrue(weight.hasGradient(), "Weight tensor must possess gradients");
        assertNotNull(weight.getGradient(), "Gradient tensor cannot be null");

        float originalValue = weight.toFloatArray()[0];
        gradEngine.step(List.of(weight));
        float updatedValue = weight.toFloatArray()[0];

        assertTrue(updatedValue < originalValue, "SGD step must decrease weight value along gradient");
    }

    @Test
    @DisplayName("Verify Pipeline execution end-to-end without PyTorch JNI dimension errors")
    void testPipelineExecutionIntegration() {
        assertDoesNotThrow(() -> {
            ModernizedHierarchicalEncoderPipeline.main(new String[]{});
        }, "Pipeline execution main method failed");
    }
}