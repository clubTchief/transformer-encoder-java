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

    // ------------------------------------------------------------------ sliding context window

    private static int argmaxOfLastPosition(NDManager manager, BlockStackLM model, List<Integer> tokens) {
        try (NDManager scope = manager.newSubManager()) {
            int len = tokens.size();
            NDArray ids = scope.create(tokens.stream().mapToInt(Integer::intValue).toArray());
            NDArray mask = CausalMaskGenerator.createCausalMask(scope, len).squeeze(0).squeeze(0);
            float[] last = model.forwardIds(scope, ids, mask).get((long) (len - 1)).toFloatArray();
            int best = 0;
            for (int k = 1; k < last.length; k++) if (last[k] > last[best]) best = k;
            return best;
        }
    }

    /** The same context-shift policy as TextGenerator, but every step recomputed from scratch without any cache. */
    private static List<Integer> greedyWithWindowByRecompute(NDManager manager, BlockStackLM model, int[] prompt, int count, int window) {
        List<Integer> cached = new ArrayList<>();
        for (int i = Math.max(0, prompt.length - (window - 1)); i < prompt.length; i++) cached.add(prompt[i]);
        List<Integer> generated = new ArrayList<>();
        int next = argmaxOfLastPosition(manager, model, cached);
        generated.add(next);
        while (generated.size() < count) {
            if (cached.size() == window) {
                cached = new ArrayList<>(cached.subList(cached.size() - window / 2, cached.size()));
            }
            cached.add(next);
            next = argmaxOfLastPosition(manager, model, cached);
            generated.add(next);
        }
        return generated;
    }

    @Test
    public void testWindowLargeEnoughChangesNothing() {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = model(manager);
            int[] prompt = {1, 2, 3};
            TextGenerator generator = new TextGenerator(model, new Sampler(1));
            List<Integer> plain = generator.generate(manager, prompt, 8, -1, 0.0f, 0, 1.0f, 1.0f);
            List<Integer> windowed = generator.generate(manager, prompt, 8, -1, 0.0f, 0, 1.0f, 1.0f, 3 + 8, null);
            Assertions.assertEquals(plain, windowed);
        }
    }

    @Test
    public void testSlidingWindowMatchesRecomputeWithTheSamePolicy() {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = model(manager);
            int[] prompt = {1, 2, 3};
            List<Integer> windowed = new TextGenerator(model, new Sampler(1))
                    .generate(manager, prompt, 30, -1, 0.0f, 0, 1.0f, 1.0f, 8, null);

            Assertions.assertEquals(30, windowed.size());
            Assertions.assertEquals(greedyWithWindowByRecompute(manager, model, prompt, 30, 8), windowed,
                    "after every context shift the cache must hold exactly what a fresh computation of that context would");
        }
    }

    @Test
    public void testPromptLongerThanTheWindowUsesItsNewestTokens() {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = model(manager);
            int[] prompt = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8};
            List<Integer> windowed = new TextGenerator(model, new Sampler(1))
                    .generate(manager, prompt, 12, -1, 0.0f, 0, 1.0f, 1.0f, 8, null);
            Assertions.assertEquals(greedyWithWindowByRecompute(manager, model, prompt, 12, 8), windowed);
        }
    }

    @Test
    public void testListenerSeesEveryTokenInOrder() {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = model(manager);
            List<Integer> heard = new ArrayList<>();
            List<Integer> result = new TextGenerator(model, new Sampler(1))
                    .generate(manager, new int[]{1, 2}, 15, -1, 0.0f, 0, 1.0f, 1.0f, 6, heard::add);
            Assertions.assertEquals(result, heard);
        }
    }

    @Test
    public void testTooSmallWindowIsRejected() {
        try (NDManager manager = NDManager.newBaseManager()) {
            TextGenerator generator = new TextGenerator(model(manager), new Sampler(1));
            Assertions.assertThrows(IllegalArgumentException.class,
                    () -> generator.generate(manager, new int[]{1}, 5, -1, 0.0f, 0, 1.0f, 1.0f, 1, null));
        }
    }

    @Test
    public void testZeroTokensIsEmptyWhateverTheWindow() {
        try (NDManager manager = NDManager.newBaseManager()) {
            TextGenerator generator = new TextGenerator(model(manager), new Sampler(1));
            Assertions.assertTrue(generator.generate(manager, new int[]{1}, 0, -1, 0.0f, 0, 1.0f, 1.0f, 1, null).isEmpty());
            Assertions.assertTrue(generator.generate(manager, new int[]{1}, -3, -1, 0.0f, 0, 1.0f, 1.0f, 2, null).isEmpty());
        }
    }
}
