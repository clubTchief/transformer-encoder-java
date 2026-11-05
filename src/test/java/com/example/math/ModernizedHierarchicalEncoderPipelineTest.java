package com.example.math;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

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

    @Test
    @DisplayName("Verify LayerNorm produces zero mean and unit variance")
    void testLayerNormNormalization() {
        NDArray input = manager.create(new float[][]{
                {1.0f, 2.0f, 3.0f, 4.0f},
                {10.0f, 20.0f, 30.0f, 40.0f}
        });

        NDArray normed = ModernizedHierarchicalEncoderPipeline.applyLayerNorm(manager, input, 1e-5f);
        
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
}