package com.example.math;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.transformer.BlockStackLM;
import com.transformer.LMConfig;
import com.transformer.ModelSerializer;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class ModelSerializationTest {

    @Test
    public void testWeightSerializationAndLoading() throws IOException {
        String path = "target/test_model_checkpoint.bin";
        try (NDManager manager = NDManager.newBaseManager()) {
            LMConfig config = new LMConfig(1, 16, 2, 32, 10, 1e-5f);

            // 1. Original model with random weights; params() is the WHOLE model (the embedding table included)
            BlockStackLM source = new BlockStackLM(manager, config);
            ModelSerializer.saveWeights(path, source.params());

            // 2. A second, fresh model with different random weights
            BlockStackLM target = new BlockStackLM(manager, config);
            ModelSerializer.loadWeights(manager, path, target.params());

            // 3. Every parameter tensor matches 1:1
            List<NDArray> sourceParams = source.params();
            List<NDArray> targetParams = target.params();
            for (int i = 0; i < sourceParams.size(); i++) {
                Assertions.assertArrayEquals(sourceParams.get(i).toFloatArray(), targetParams.get(i).toFloatArray(), 1e-6f,
                        "Parameter tensor mismatch at index " + i);
            }
            new File(path).deleteOnExit();
        }
    }

    @Test
    public void testLoadedModelProducesIdenticalLogits() throws IOException {
        String path = "target/test_model_checkpoint_logits.bin";
        try (NDManager manager = NDManager.newBaseManager()) {
            LMConfig config = new LMConfig(2, 16, 2, 32, 10, 1e-5f);
            BlockStackLM source = new BlockStackLM(manager, config);
            ModelSerializer.saveWeights(path, source.params());
            BlockStackLM target = new BlockStackLM(manager, config);
            ModelSerializer.loadWeights(manager, path, target.params());

            NDArray ids = manager.create(new int[]{1, 2, 3, 4});
            NDArray mask = com.transformer.CausalMaskGenerator.createCausalMask(manager, 4).squeeze(0).squeeze(0);
            Assertions.assertArrayEquals(source.forwardIds(manager, ids, mask).toFloatArray(),
                    target.forwardIds(manager, ids, mask).toFloatArray(), 1e-6f);
            new File(path).deleteOnExit();
        }
    }

    @Test
    public void testCheckpointOfAnotherArchitectureIsRejected() throws IOException {
        String path = "target/test_model_checkpoint_mismatch.bin";
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM small = new BlockStackLM(manager, new LMConfig(1, 16, 2, 32, 10, 1e-5f));
            ModelSerializer.saveWeights(path, small.params());

            // same number of tensors, different shapes: this used to be loaded without any complaint
            BlockStackLM wider = new BlockStackLM(manager, new LMConfig(1, 32, 2, 32, 10, 1e-5f));
            Assertions.assertThrows(IllegalArgumentException.class,
                    () -> ModelSerializer.loadWeights(manager, path, wider.params()));
            new File(path).deleteOnExit();
        }
    }
}
