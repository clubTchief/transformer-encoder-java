package com.generation;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.transformer.BlockStackLM;
import com.transformer.CausalMaskGenerator;
import com.transformer.LMConfig;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

public class TextGeneratorTest {

    private static final int VOCAB = 13;

    private static BlockStackLM model(NDManager manager) {
        Engine.getInstance().setRandomSeed(5);
        return new BlockStackLM(manager, new LMConfig(2, 16, 2, 32, VOCAB, 1e-5f));
    }

    /** The slow, obviously correct reference: re-run the whole sequence for every new token. */
    private static List<Integer> greedyByFullRecompute(NDManager manager, BlockStackLM model, int[] prompt, int count) {
        List<Integer> ids = new ArrayList<>();
        for (int p : prompt) ids.add(p);
        List<Integer> generated = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            try (NDManager scope = manager.newSubManager()) {
                int len = ids.size();
                NDArray idArray = scope.create(ids.stream().mapToInt(Integer::intValue).toArray());
                NDArray mask = CausalMaskGenerator.createCausalMask(scope, len).squeeze(0).squeeze(0);
                float[] last = model.forwardIds(scope, idArray, mask).get((long) (len - 1)).toFloatArray();
                int best = 0;
                for (int k = 1; k < last.length; k++) if (last[k] > last[best]) best = k;
                ids.add(best);
                generated.add(best);
            }
        }
        return generated;
    }

    @Test
    public void testGreedyCachedGenerationMatchesFullRecompute() {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = model(manager);
            int[] prompt = {1, 2, 3};

            List<Integer> cached = new TextGenerator(model, new Sampler(1))
                    .generate(manager, prompt, 8, -1, 0.0f, 0, 1.0f, 1.0f);

            // Equality with the reference proves that every sampled token is embedded and fed back at the right position
            Assertions.assertEquals(greedyByFullRecompute(manager, model, prompt, 8), cached);
        }
    }

    @Test
    public void testGreedyGenerationIsDeterministic() {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = model(manager);
            TextGenerator generator = new TextGenerator(model, new Sampler(1));
            List<Integer> first = generator.generate(manager, new int[]{4, 5}, 6, -1, 0.0f, 0, 1.0f, 1.0f);
            List<Integer> second = generator.generate(manager, new int[]{4, 5}, 6, -1, 0.0f, 0, 1.0f, 1.0f);
            Assertions.assertEquals(first, second);
        }
    }

    @Test
    public void testStopsAfterEndOfSequenceToken() {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = model(manager);
            TextGenerator generator = new TextGenerator(model, new Sampler(1));
            int[] prompt = {1, 2, 3};
            int firstToken = generator.generate(manager, prompt, 1, -1, 0.0f, 0, 1.0f, 1.0f).get(0);

            List<Integer> stopped = generator.generate(manager, prompt, 10, firstToken, 0.0f, 0, 1.0f, 1.0f);
            Assertions.assertEquals(List.of(firstToken), stopped, "the EOS token is included and generation ends there");
        }
    }

    @Test
    public void testRespectsMaxGenTokens() {
        try (NDManager manager = NDManager.newBaseManager()) {
            TextGenerator generator = new TextGenerator(model(manager), new Sampler(1));
            Assertions.assertEquals(5, generator.generate(manager, new int[]{1}, 5, -1, 0.8f, 5, 0.9f, 1.1f).size());
            Assertions.assertTrue(generator.generate(manager, new int[]{1}, 0, -1, 0.8f, 5, 0.9f, 1.1f).isEmpty());
        }
    }

    @Test
    public void testSeededSamplingIsReproducible() {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = model(manager);
            List<Integer> a = new TextGenerator(model, new Sampler(7)).generate(manager, new int[]{1, 2}, 8, -1, 1.0f, 0, 1.0f, 1.0f);
            List<Integer> b = new TextGenerator(model, new Sampler(7)).generate(manager, new int[]{1, 2}, 8, -1, 1.0f, 0, 1.0f, 1.0f);
            Assertions.assertEquals(a, b);
        }
    }

    @Test
    public void testRejectsEmptyPrompt() {
        try (NDManager manager = NDManager.newBaseManager()) {
            TextGenerator generator = new TextGenerator(model(manager), new Sampler(1));
            Assertions.assertThrows(IllegalArgumentException.class,
                    () -> generator.generate(manager, new int[0], 5, -1, 0.0f, 0, 1.0f, 1.0f));
        }
    }
}
