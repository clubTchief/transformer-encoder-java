package com.example.math;

import com.transformer.ModelSerializer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ModelSerializationTest {

    @Test
    public void testWeightSerializationAndLoading() throws IOException {
        String testCheckpointPath = "target/test_model_checkpoint.bin";

        try (NDManager manager = NDManager.newBaseManager()) {
            int dModel = 16;
            int numHeads = 2;
            int numLayers = 1;
            int dFF = 32;
            int vocabSize = 10;
            float eps = 1e-5f;

            // 1. Initialize original model & embedding table with random weights
            BlockStackLM sourceModel = new BlockStackLM(manager, dModel, numHeads, numLayers, dFF, vocabSize, eps);
            NDArray sourceEmbedding = manager.randomNormal(0.0f, 1.0f, new Shape(vocabSize, dModel), DataType.FLOAT32);

            List<NDArray> sourceParams = new ArrayList<>(sourceModel.params());
            sourceParams.add(sourceEmbedding);

            // Save source parameters to disk
            ModelSerializer.saveWeights(testCheckpointPath, sourceParams);

            // 2. Instantiate a second, fresh model with different initial weights
            BlockStackLM targetModel = new BlockStackLM(manager, dModel, numHeads, numLayers, dFF, vocabSize, eps);
            NDArray targetEmbedding = manager.zeros(new Shape(vocabSize, dModel), DataType.FLOAT32);

            List<NDArray> targetParams = new ArrayList<>(targetModel.params());
            targetParams.add(targetEmbedding);

            // Load saved checkpoint weights into target model
            ModelSerializer.loadWeights(manager, testCheckpointPath, targetParams);

            // 3. Verify float array equality across all parameter tensors
            for (int i = 0; i < sourceParams.size(); i++) {
                float[] originalVals = sourceParams.get(i).toFloatArray();
                float[] loadedVals = targetParams.get(i).toFloatArray();

                Assertions.assertArrayEquals(originalVals, loadedVals, 1e-6f, 
                        "Parameter tensor mismatch at index " + i);
            }

            System.out.println("✅ Weight serialization test passed: Loaded parameters match original trained weights 1:1.");
            new File(testCheckpointPath).deleteOnExit();
        }
    }
}