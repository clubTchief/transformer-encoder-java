package com.example.math;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MultiEpochTrainer {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {

            // Model Configuration Hyperparameters
            int dModel = 64;
            int numHeads = 4;
            int numLayers = 2;
            int dFF = 256;
            float eps = 1e-5f;
            float learningRate = 0.05f; // Increased slightly for visible convergence
            int epochs = 50;

            // Target Training Story Sequence
            String[] targetWords = {
                "Once", "upon", "a", "time", "in", "a", "distant", "realm", "lived", "a", "brave", "knight", "."
            };
            int seqLen = targetWords.length;

            // Build Token-to-ID Vocabulary Mapping
            Map<String, Integer> tokenToId = new HashMap<>();
            Map<Integer, String> idToToken = new HashMap<>();
            for (String word : targetWords) {
                if (!tokenToId.containsKey(word)) {
                    int id = tokenToId.size();
                    tokenToId.put(word, id);
                    idToToken.put(id, word);
                }
            }
            int vocabSize = tokenToId.size();

            // Map target word sequence to integer token IDs
            int[] targetTokenIds = new int[seqLen];
            for (int i = 0; i < seqLen; i++) {
                targetTokenIds[i] = tokenToId.get(targetWords[i]);
            }

            // Initialize Model Stack and Gradient Engine
            BlockStackLM model = new BlockStackLM(manager, dModel, numHeads, numLayers, dFF, vocabSize, eps);
            GradientEngine gradEngine = new GradientEngine(learningRate);

            // Initialize Input Embedding Table [vocabSize, dModel]
            NDArray embeddingTable = manager.randomNormal(0.0f, 0.02f, new Shape(vocabSize, dModel), DataType.FLOAT32);
            embeddingTable.setRequiresGradient(true);

            // Register all parameter lists (Decoder blocks + LM Head + Embedding Table)
            List<NDArray> fullTrainableParams = new ArrayList<>(model.params());
            fullTrainableParams.add(embeddingTable);

            // Create 2D Causal Mask for Attention [seqLen, seqLen]
            NDArray causalMask4D = CausalMaskGenerator.createCausalMask(manager, seqLen);
            NDArray causalMask = causalMask4D.squeeze(0).squeeze(0);

            System.out.println("================================================================================");
            System.out.println("            >>> STARTING MULTI-EPOCH TRAINING LOOP <<<                          ");
            System.out.println("================================================================================");
            System.out.printf("Sequence Length: %d | Vocab Size: %d | Parameters: %d tensors | Epochs: %d\n",
                    seqLen, vocabSize, fullTrainableParams.size(), epochs);
            System.out.println("--------------------------------------------------------------------------------");

            float initialLoss = 0.0f;
            float finalLoss = 0.0f;

            // Multi-Epoch Optimization Loop
            for (int epoch = 1; epoch <= epochs; epoch++) {
                float epochLoss;

                // Open autograd tape before slice lookup and forward pass
                try (GradientCollector collector = gradEngine.createCollector()) {

                    // 1. Rebuild inputX dynamically so indexing operations are tracked on autograd graph
                    NDArray inputX = manager.create(new Shape(seqLen, dModel), DataType.FLOAT32);
                    for (int i = 0; i < seqLen; i++) {
                        NDArray tokenVec = embeddingTable.get(targetTokenIds[i]);
                        inputX.set(new ai.djl.ndarray.index.NDIndex("{}:{}", i, i + 1), tokenVec);
                    }

                    // 2. Forward pass to produce logits
                    NDArray logits = model.forward(manager, inputX, causalMask);

                    // 3. Compute loss directly as a connected NDArray
                    NDArray lossTensor = computeDifferentiableLoss(manager, logits, targetTokenIds);
                    epochLoss = lossTensor.toFloatArray()[0];

                    // 4. Trigger backward flow through the active tape
                    collector.backward(lossTensor);
                }

                if (epoch == 1) initialLoss = epochLoss;
                finalLoss = epochLoss;

                // Apply SGD parameter updates and clear gradient buffers
                gradEngine.step(fullTrainableParams);

                // Print loss metrics periodically
                if (epoch == 1 || epoch % 10 == 0 || epoch == epochs) {
                    System.out.printf("Epoch %3d/%3d | Loss: %.6f\n", epoch, epochs, epochLoss);
                }
            }

            System.out.println("--------------------------------------------------------------------------------");
            System.out.printf("Initial Loss: %.6f | Final Loss: %.6f | Total Loss Reduction: %.2f%%\n",
                    initialLoss, finalLoss, ((initialLoss - finalLoss) / initialLoss) * 100.0f);
            System.out.println("================================================================================");
        }
    }

    /**
     * Differentiable loss helper that preserves the computation graph for DJL collector.backward().
     */
    private static NDArray computeDifferentiableLoss(NDManager manager, NDArray logits, int[] targetTokenIds) {
        long seqLen = logits.getShape().get(0);
        long vocabSize = logits.getShape().get(1);

        // Compute Log-Softmax across vocabulary dimension
        NDArray maxLogits = logits.max(new int[]{1}, true);
        NDArray shiftedLogits = logits.sub(maxLogits);
        NDArray expLogits = shiftedLogits.exp();
        NDArray sumExp = expLogits.sum(new int[]{1}, true);
        NDArray logSoftmax = shiftedLogits.sub(sumExp.log());

        // Gather target token log-probabilities
        NDArray totalLoss = manager.create(0.0f);
        for (int i = 0; i < seqLen; i++) {
            int targetId = targetTokenIds[i];
            NDArray targetLogProb = logSoftmax.get(new ai.djl.ndarray.index.NDIndex("{}, {}", i, targetId));
            totalLoss = totalLoss.sub(targetLogProb);
        }

        return totalLoss.div((float) seqLen);
    }
}