package com.dictionary;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import com.transformer.RMSNorm;
import com.transformer.BlockStackLM;
import com.transformer.CausalLossCalculator;
import com.transformer.CausalMaskGenerator;
import com.transformer.AdamOptimizer;
import com.transformer.ModelSerializer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MultiBatchTrainer {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {

            List<String> corpus = Arrays.asList(
                "Once upon a time in a distant realm lived a brave knight .",
                "The knight found a mysterious glowing sword inside a dark cave .",
                "A powerful dragon protected the ancient castle in the realm .",
                "The brave knight defended the realm and restored peace to all ."
            );

            int seqLen = 12;
            int batchSize = 2;
            int epochs = 100;
            float learningRate = 0.01f;

            Vocabulary vocab = new Vocabulary(corpus);
            StoryDataset dataset = new StoryDataset(corpus, vocab, seqLen);

            System.out.println("================================================================================");
            System.out.println("          >>> INITIALIZING MULTI-BATCH TRANSFORMER TRAINER <<<                 ");
            System.out.println("================================================================================");
            System.out.println("Corpus Samples : " + corpus.size());
            System.out.println("Vocabulary Size: " + vocab.size());
            System.out.println("Total Samples  : " + dataset.size());
            System.out.println("Batch Size     : " + batchSize + " | Batches per Epoch: " + dataset.getNumBatches(batchSize));
            System.out.println("--------------------------------------------------------------------------------");

            int dModel = 64;
            int numHeads = 4;
            int numLayers = 2;
            int dFF = 256;
            float eps = 1e-5f;

            BlockStackLM model = new BlockStackLM(manager, dModel, numHeads, numLayers, dFF, vocab.size(), eps);
            NDArray embeddingTable = manager.randomUniform(-0.1f, 0.1f, new Shape(vocab.size(), dModel), DataType.FLOAT32);

            List<NDArray> trainableParams = new ArrayList<>(model.params());
            trainableParams.add(embeddingTable);

            for (NDArray param : trainableParams) {
                param.setRequiresGradient(true);
            }

            // Move embedding table data to a Java float array for O(1) host lookup
            float[] embedFlat = embeddingTable.toFloatArray();

            // Pre-allocate static causal mask (reusable across all batches)
            NDArray causalMask = CausalMaskGenerator.createCausalMask(manager, seqLen).squeeze(0).squeeze(0);

            AdamOptimizer optimizer = new AdamOptimizer(trainableParams, learningRate, 0.9f, 0.999f, 1e-8f);

            for (int epoch = 1; epoch <= epochs; epoch++) {
                float totalEpochLoss = 0.0f;
                int numBatches = dataset.getNumBatches(batchSize);

                for (int b = 0; b < numBatches; b++) {
                    StoryDataset.Batch batch = dataset.getBatch(manager, b, batchSize);
                    long currentBatchSize = batch.inputs.getShape().get(0);

                    // Extract input batch IDs to Java memory in ONE native call
                    float[] inputIds = batch.inputs.toFloatArray();
                    float[] batchEmbedData = new float[(int) (currentBatchSize * seqLen * dModel)];

                    // Vectorized embedding reconstruction
                    for (int i = 0; i < currentBatchSize; i++) {
                        for (int t = 0; t < seqLen; t++) {
                            int tokenId = (int) inputIds[i * seqLen + t];
                            int srcPos = tokenId * dModel;
                            int dstPos = (int) ((i * seqLen + t) * dModel);
                            System.arraycopy(embedFlat, srcPos, batchEmbedData, dstPos, dModel);
                        }
                    }

                    try (NDManager childManager = manager.newSubManager();
                         GradientCollector gc = childManager.getEngine().newGradientCollector()) {

                        NDArray batchInputX = childManager.create(batchEmbedData).reshape(new Shape(currentBatchSize, seqLen, dModel));

                        // Forward pass & Causal Loss
                        NDArray logits = model.forward(childManager, batchInputX, causalMask);
                        NDArray loss = CausalLossCalculator.calculateLoss(childManager, logits, batch.targets, vocab.getPadId());
                        
                        gc.backward(loss);
                        totalEpochLoss += loss.toFloatArray()[0];
                    }

                    optimizer.step();
                    
                    // Sync refreshed weight cache after optimizer step
                    embedFlat = embeddingTable.toFloatArray();
                }

                float avgLoss = totalEpochLoss / numBatches;

                if (epoch % 10 == 0 || epoch == 1) {
                    System.out.printf("Epoch %3d/%d | Average Loss: %.6f\n", epoch, epochs, avgLoss);
                }
            }

            // Ensure output directory exists before serialization
            File modelsDir = new File("models");
            if (!modelsDir.exists()) {
                modelsDir.mkdirs();
            }

            String checkpointPath = "models/story_model.bin";
            ModelSerializer.saveWeights(checkpointPath, trainableParams);
            System.out.println("Successfully serialized trained weights to: " + checkpointPath);
            System.out.println("================================================================================");

        } catch (IOException e) {
            System.err.println("Failed to save checkpoint: " + e.getMessage());
        }
    }
}