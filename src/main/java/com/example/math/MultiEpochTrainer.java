package com.example.math;

import com.transformer.ModelSerializer;
import com.transformer.GradientEngine;
import com.transformer.CausalMaskGenerator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;

import java.io.IOException;
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
            float learningRate = 0.05f;
            int epochs = 60;

            // Target Training Story Sequence
            String[] targetWords = {
                "Once", "upon", "a", "time", "in", "a", "distant", "realm", "lived", "a", "brave", "knight", "."
            };
            int fullLen = targetWords.length;
            int trainLen = fullLen - 1; // Input length = total tokens - 1

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

            int[] fullTokenIds = new int[fullLen];
            for (int i = 0; i < fullLen; i++) {
                fullTokenIds[i] = tokenToId.get(targetWords[i]);
            }

            // Input token IDs (0 to N-2) & Target token IDs shifted by 1 (1 to N-1)
            int[] inputTokenIds = new int[trainLen];
            int[] targetTokenIds = new int[trainLen];
            for (int i = 0; i < trainLen; i++) {
                inputTokenIds[i] = fullTokenIds[i];
                targetTokenIds[i] = fullTokenIds[i + 1];
            }

            // Initialize Model Stack and Gradient Engine
            BlockStackLM model = new BlockStackLM(manager, dModel, numHeads, numLayers, dFF, vocabSize, eps);
            GradientEngine gradEngine = new GradientEngine(learningRate);

            // Initialize Input Embedding Table [vocabSize, dModel]
            NDArray embeddingTable = manager.randomNormal(0.0f, 0.02f, new Shape(vocabSize, dModel), DataType.FLOAT32);
            embeddingTable.setRequiresGradient(true);

            List<NDArray> fullTrainableParams = new ArrayList<>(model.params());
            fullTrainableParams.add(embeddingTable);

            // Create 2D Causal Mask for Attention [trainLen, trainLen]
            NDArray causalMask4D = CausalMaskGenerator.createCausalMask(manager, trainLen);
            NDArray causalMask = causalMask4D.squeeze(0).squeeze(0);

            System.out.println("================================================================================");
            System.out.println("            >>> STARTING MULTI-EPOCH TRAINING LOOP <<<                          ");
            System.out.println("================================================================================");
            System.out.printf("Training Sequence Length: %d | Vocab Size: %d | Epochs: %d\n",
                    trainLen, vocabSize, epochs);
            System.out.println("--------------------------------------------------------------------------------");

            float initialLoss = 0.0f;
            float finalLoss = 0.0f;

            // Multi-Epoch Optimization Loop
            for (int epoch = 1; epoch <= epochs; epoch++) {
                float epochLoss;

                try (GradientCollector collector = gradEngine.createCollector()) {

                    // 1. Build inputX dynamically for tokens 0..N-2
                    NDArray inputX = manager.create(new Shape(trainLen, dModel), DataType.FLOAT32);
                    for (int i = 0; i < trainLen; i++) {
                        NDArray tokenVec = embeddingTable.get(inputTokenIds[i]);
                        inputX.set(new ai.djl.ndarray.index.NDIndex("{}:{}", i, i + 1), tokenVec);
                    }

                    // 2. Forward pass
                    NDArray logits = model.forward(manager, inputX, causalMask);

                    // 3. Loss calculation predicting tokens 1..N-1
                    NDArray lossTensor = computeDifferentiableLoss(manager, logits, targetTokenIds);
                    epochLoss = lossTensor.toFloatArray()[0];

                    // 4. Backward pass
                    collector.backward(lossTensor);
                }

                if (epoch == 1) initialLoss = epochLoss;
                finalLoss = epochLoss;

                gradEngine.step(fullTrainableParams);

                if (epoch == 1 || epoch % 10 == 0 || epoch == epochs) {
                    System.out.printf("Epoch %3d/%3d | Loss: %.6f\n", epoch, epochs, epochLoss);
                }
            }

            System.out.println("--------------------------------------------------------------------------------");
            System.out.printf("Initial Loss: %.6f | Final Loss: %.6f | Total Loss Reduction: %.2f%%\n",
                    initialLoss, finalLoss, ((initialLoss - finalLoss) / initialLoss) * 100.0f);
            System.out.println("================================================================================");

            // ================================================================================
            //            >>> SERIALIZING TRAINED MODEL WEIGHTS <<<
            // ================================================================================
            String checkpointPath = "models/story_model.bin";
            try {
                ModelSerializer.saveWeights(checkpointPath, fullTrainableParams);
            } catch (IOException e) {
                System.err.println("Failed to serialize model parameters: " + e.getMessage());
            }

            // ================================================================================
            //            >>> GENERATING STORY FROM TRAINED WEIGHTS <<<                       
            // ================================================================================
            System.out.println("\n=== GENERATING STORY FROM TRAINED WEIGHTS ===");

            List<Integer> generatedIds = new ArrayList<>();
            generatedIds.add(tokenToId.get("Once"));

            int maxTokensToGenerate = trainLen;
            System.out.print("Generated Text: " + idToToken.get(generatedIds.get(0)) + " ");

            for (int step = 0; step < maxTokensToGenerate; step++) {
                int currentLen = generatedIds.size();

                NDArray genInputX = manager.create(new Shape(currentLen, dModel), DataType.FLOAT32);
                for (int i = 0; i < currentLen; i++) {
                    NDArray tokenVec = embeddingTable.get(generatedIds.get(i));
                    genInputX.set(new ai.djl.ndarray.index.NDIndex("{}:{}", i, i + 1), tokenVec);
                }

                NDArray genMask4D = CausalMaskGenerator.createCausalMask(manager, currentLen);
                NDArray genMask = genMask4D.squeeze(0).squeeze(0);
                NDArray logits = model.forward(manager, genInputX, genMask);

                NDArray lastLogits = logits.get((long) (currentLen - 1));

                float[] logitVals = lastLogits.toFloatArray();
                int bestTokenId = 0;
                float maxVal = Float.NEGATIVE_INFINITY;
                for (int v = 0; v < logitVals.length; v++) {
                    if (logitVals[v] > maxVal) {
                        maxVal = logitVals[v];
                        bestTokenId = v;
                    }
                }

                generatedIds.add(bestTokenId);
                String word = idToToken.get(bestTokenId);
                System.out.print(word + " ");

                if (".".equals(word)) break;
            }

            System.out.println("\n================================================================================\n");
        }
    }

    private static NDArray computeDifferentiableLoss(NDManager manager, NDArray logits, int[] targetTokenIds) {
        long seqLen = logits.getShape().get(0);

        NDArray maxLogits = logits.max(new int[]{1}, true);
        NDArray shiftedLogits = logits.sub(maxLogits);
        NDArray expLogits = shiftedLogits.exp();
        NDArray sumExp = expLogits.sum(new int[]{1}, true);
        NDArray logSoftmax = shiftedLogits.sub(sumExp.log());

        NDArray totalLoss = manager.create(0.0f);
        for (int i = 0; i < seqLen; i++) {
            int targetId = targetTokenIds[i];
            NDArray targetLogProb = logSoftmax.get(new ai.djl.ndarray.index.NDIndex("{}, {}", i, targetId));
            totalLoss = totalLoss.sub(targetLogProb);
        }

        return totalLoss.div((float) seqLen);
    }
}