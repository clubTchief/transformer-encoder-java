package com.example.math;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class GradientFlowVerificationTest {

    @Test
    @DisplayName("Verify gradient flow reaches every parameter and gradients are non-zero")
    void testGradientFlowReachesAllWeights() {
        try (NDManager manager = NDManager.newBaseManager()) {
            int seqLen = 4;
            int dModel = 384;
            int numHeads = 8;
            int dFF = 4 * dModel;
            int dK = dModel / numHeads;
            float eps = 1e-5f;

            GradientEngine gradEngine = new GradientEngine(0.001f);
            List<NDArray> registeredParameters = new ArrayList<>();

            // 1. Initialize Inputs (Simulated sequence embeddings X_0)
            NDArray currentX = manager.randomNormal(0.0f, 1.0f, new Shape(seqLen, dModel), DataType.FLOAT32);
            currentX.setRequiresGradient(true); // Track input gradient as reference

            // 2. Initialize LayerNorm components
            LayerNorm norm1 = new LayerNorm(manager, dModel, eps);
            LayerNorm norm2 = new LayerNorm(manager, dModel, eps);

            registeredParameters.add(norm1.getGamma());
            registeredParameters.add(norm1.getBeta());
            registeredParameters.add(norm2.getGamma());
            registeredParameters.add(norm2.getBeta());

            // 3. Initialize Attention Projection Weights (Q, K, V, Out)
            float stddevK = (float) Math.sqrt(2.0 / (dModel + dK));
            float stddevO = (float) Math.sqrt(2.0 / (dModel + dModel));

            NDArray W_Q = manager.randomNormal(0.0f, stddevK, new Shape(dModel, dModel), DataType.FLOAT32);
            NDArray W_K = manager.randomNormal(0.0f, stddevK, new Shape(dModel, dModel), DataType.FLOAT32);
            NDArray W_V = manager.randomNormal(0.0f, stddevK, new Shape(dModel, dModel), DataType.FLOAT32);
            NDArray W_O = manager.randomNormal(0.0f, stddevO, new Shape(dModel, dModel), DataType.FLOAT32);

            W_Q.setRequiresGradient(true);
            W_K.setRequiresGradient(true);
            W_V.setRequiresGradient(true);
            W_O.setRequiresGradient(true);

            registeredParameters.add(W_Q);
            registeredParameters.add(W_K);
            registeredParameters.add(W_V);
            registeredParameters.add(W_O);

            // 4. Initialize SwiGLU FFN Weights & Biases
            float stddev1 = (float) Math.sqrt(2.0 / (dModel + dFF));
            float stddev2 = (float) Math.sqrt(2.0 / (dFF + dModel));

            NDArray W_gate = manager.randomNormal(0.0f, stddev1, new Shape(dModel, dFF), DataType.FLOAT32);
            NDArray b_gate = manager.zeros(new Shape(dFF), DataType.FLOAT32);
            NDArray W_1 = manager.randomNormal(0.0f, stddev1, new Shape(dModel, dFF), DataType.FLOAT32);
            NDArray b_1 = manager.zeros(new Shape(dFF), DataType.FLOAT32);
            NDArray W_2 = manager.randomNormal(0.0f, stddev2, new Shape(dFF, dModel), DataType.FLOAT32);
            NDArray b_2 = manager.zeros(new Shape(dModel), DataType.FLOAT32);

            W_gate.setRequiresGradient(true);
            b_gate.setRequiresGradient(true);
            W_1.setRequiresGradient(true);
            b_1.setRequiresGradient(true);
            W_2.setRequiresGradient(true);
            b_2.setRequiresGradient(true);

            registeredParameters.add(W_gate);
            registeredParameters.add(b_gate);
            registeredParameters.add(W_1);
            registeredParameters.add(b_1);
            registeredParameters.add(W_2);
            registeredParameters.add(b_2);

            // 5. Generate Causal Mask
            NDArray causalMask4D = CausalMaskGenerator.createCausalMask(manager, seqLen);
            NDArray causalMask = causalMask4D.squeeze(0).squeeze(0);

            // -------------------------------------------------------------------------
            // FORWARD + BACKWARD PASS UNDER GRADIENT COLLECTOR
            // -------------------------------------------------------------------------
            try (GradientCollector collector = gradEngine.createCollector()) {
                // SubLayer 1: Pre-LN + Causal MHA with Differentiable RoPE
                NDArray normX1 = norm1.forward(currentX);

                NDArray Q = normX1.matMul(W_Q);
                NDArray K = normX1.matMul(W_K);
                NDArray V = normX1.matMul(W_V);

                NDArray Q_split = Q.reshape(new Shape(seqLen, numHeads, dK)).transpose(1, 0, 2);
                NDArray K_split = K.reshape(new Shape(seqLen, numHeads, dK)).transpose(1, 0, 2);
                NDArray V_split = V.reshape(new Shape(seqLen, numHeads, dK)).transpose(1, 0, 2);

                // Fully differentiable RoPE rotation
                NDArray Q_rope = ModernizedHierarchicalEncoderPipeline.applyRoPE(manager, Q_split);
                NDArray K_rope = ModernizedHierarchicalEncoderPipeline.applyRoPE(manager, K_split);

                NDArray scores = Q_rope.matMul(K_rope.transpose(0, 2, 1)).div(Math.sqrt(dK));
                NDArray maskedScores = scores.add(causalMask);
                NDArray attnWeights = maskedScores.softmax(-1);

                NDArray headOutputs = attnWeights.matMul(V_split);
                NDArray concatOutputs = headOutputs.transpose(1, 0, 2).reshape(new Shape(seqLen, dModel));
                NDArray mhaOutput = concatOutputs.matMul(W_O);

                NDArray subLayer1Out = currentX.add(mhaOutput);

                // SubLayer 2: Pre-LN + SwiGLU FFN
                NDArray normX2 = norm2.forward(subLayer1Out);

                NDArray gateLinear = normX2.matMul(W_gate).add(b_gate);
                NDArray gateSwish = ModernizedHierarchicalEncoderPipeline.applySwish(gateLinear);
                NDArray featureLinear = normX2.matMul(W_1).add(b_1);
                NDArray ffnHidden = gateSwish.mul(featureLinear);
                NDArray ffnOutput = ffnHidden.matMul(W_2).add(b_2);

                NDArray layerOutput = subLayer1Out.add(ffnOutput);

                // Compute Scalar Loss (Sum over all outputs)
                NDArray loss = layerOutput.sum();

                // Trigger Backpropagation
                collector.backward(loss);
            }

            // -------------------------------------------------------------------------
            // AUDIT & ASSERT GRADIENTS FOR ALL PARAMETERS
            // -------------------------------------------------------------------------
            System.out.println("==========================================================================");
            System.out.println("                   GRADIENT FLOW AUDIT & ASSERTIONS                      ");
            System.out.println("==========================================================================");

            for (int i = 0; i < registeredParameters.size(); i++) {
                NDArray param = registeredParameters.get(i);

                // Assertion 1: Gradient tensor exists
                assertTrue(param.hasGradient(), "Parameter at index " + i + " must have a gradient registration.");
                NDArray grad = param.getGradient();
                assertNotNull(grad, "Gradient tensor at index " + i + " cannot be null.");

                // Assertion 2: Gradient shape matches parameter shape
                assertEquals(param.getShape(), grad.getShape(), "Gradient shape must match parameter shape.");

                // Assertion 3: Gradient is NOT all zeros
                float[] gradValues = grad.toFloatArray();
                boolean hasNonZero = false;
                double sumAbsGrad = 0.0;

                for (float val : gradValues) {
                    if (Math.abs(val) > 1e-9f) {
                        hasNonZero = true;
                    }
                    sumAbsGrad += Math.abs(val);
                }

                System.out.printf("Param Index %-2d | Shape: %-14s | L1 Grad Norm: %-12.6f | Flow Status: PASS%n",
                        i, param.getShape().toString(), sumAbsGrad);

                assertTrue(hasNonZero, "Gradient for parameter at index " + i + " (Shape: " + param.getShape() + ") is all zeros!");
            }
            System.out.println("==========================================================================");
        }
    }
}