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

class DecoderBlockTest {

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
    @DisplayName("Verify DecoderBlock forward pass executes cleanly and preserves shape")
    void testForwardPassShapePreservation() {
        int seqLen = 4;
        int dModel = 384;
        int numHeads = 8;
        int dFF = 4 * dModel;
        int dK = dModel / numHeads;
        float eps = 1e-5f;

        // 1. Create DecoderBlock instance
        DecoderBlock block = new DecoderBlock(manager, dModel, numHeads, dFF, eps);

        // 2. Prepare dummy input X [seqLen, dModel]
        NDArray X = manager.randomNormal(0.0f, 1.0f, new Shape(seqLen, dModel), DataType.FLOAT32);

        // 3. Prepare 2D causal mask [seqLen, seqLen]
        NDArray causalMask4D = CausalMaskGenerator.createCausalMask(manager, seqLen);
        NDArray causalMask = causalMask4D.squeeze(0).squeeze(0);

        // 4. Pre-compute RoPE tables using correct pipeline method
        NDArray[] ropeTables = ModernizedHierarchicalEncoderPipeline.precomputeRoPEFrequencies(manager, seqLen, dK);
        NDArray cos = ropeTables[0];
        NDArray sin = ropeTables[1];

        // 5. Run forward pass
        NDArray output = block.forward(X, causalMask, cos, sin);

        assertNotNull(output);
        assertEquals(new Shape(seqLen, dModel), output.getShape(), "DecoderBlock forward must preserve input tensor shape");
    }

    @Test
    @DisplayName("Verify DecoderBlock registers all 14 trainable parameters with gradient tracking")
    void testParameterRegistration() {
        int dModel = 384;
        int numHeads = 8;
        int dFF = 1536;
        float eps = 1e-5f;

        DecoderBlock block = new DecoderBlock(manager, dModel, numHeads, dFF, eps);
        List<NDArray> params = block.params();

        // 2 LayerNorms x 2 params (gamma, beta) + 4 MHA weights + 6 FFN weights/biases = 14 params
        assertEquals(14, params.size(), "DecoderBlock must export 14 trainable parameters");

        for (int i = 0; i < params.size(); i++) {
            NDArray param = params.get(i);
            assertNotNull(param, "Parameter at index " + i + " cannot be null");
            assertTrue(param.hasGradient(), "Parameter at index " + i + " must have gradients enabled");
        }
    }

    @Test
    @DisplayName("Verify autograd gradient flow reaches every parameter in DecoderBlock")
    void testGradientFlowThroughBlock() {
        int seqLen = 4;
        int dModel = 384;
        int numHeads = 8;
        int dFF = 4 * dModel;
        int dK = dModel / numHeads;

        DecoderBlock block = new DecoderBlock(manager, dModel, numHeads, dFF, 1e-5f);
        GradientEngine gradEngine = new GradientEngine(0.001f);

        NDArray X = manager.randomNormal(0.0f, 1.0f, new Shape(seqLen, dModel), DataType.FLOAT32);
        NDArray causalMask = CausalMaskGenerator.createCausalMask(manager, seqLen).squeeze(0).squeeze(0);
        NDArray[] ropeTables = ModernizedHierarchicalEncoderPipeline.precomputeRoPEFrequencies(manager, seqLen, dK);

        try (GradientCollector collector = gradEngine.createCollector()) {
            NDArray output = block.forward(X, causalMask, ropeTables[0], ropeTables[1]);
            NDArray loss = output.sum();
            collector.backward(loss);
        }

        // Verify every parameter receives non-null gradient after backprop
        for (NDArray param : block.params()) {
            assertNotNull(param.getGradient(), "Gradient cannot be null for parameter shape: " + param.getShape());
            
            // Check for non-zero gradient values
            float[] gradValues = param.getGradient().toFloatArray();
            boolean nonZero = false;
            for (float val : gradValues) {
                if (Math.abs(val) > 1e-9f) {
                    nonZero = true;
                    break;
                }
            }
            assertTrue(nonZero, "Gradient for parameter shape " + param.getShape() + " should not be all zeros");
        }
    }

    @Test
    @DisplayName("Verify DecoderBlock.Trace wrapper holds output correctly")
    void testTraceInnerClass() {
        NDArray dummyOutput = manager.ones(new Shape(4, 384));
        DecoderBlock.Trace trace = new DecoderBlock.Trace(dummyOutput);

        assertNotNull(trace.getOutput());
        assertEquals(new Shape(4, 384), trace.getOutput().getShape());
    }
}