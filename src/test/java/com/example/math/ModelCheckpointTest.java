package com.example.math;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.example.story.CharTokenizer;
import com.transformer.BlockStackLM;
import com.transformer.CausalMaskGenerator;
import com.transformer.LMConfig;
import com.transformer.ModelCheckpoint;
import com.transformer.ModelSerializer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ModelCheckpointTest {

    private static final CharTokenizer TOKENIZER = CharTokenizer.fromText("abcabcabc xyz", 1);

    private static BlockStackLM model(NDManager manager) {
        Engine.getInstance().setRandomSeed(3);
        return new BlockStackLM(manager, new LMConfig(2, 16, 2, 32, TOKENIZER.vocabSize(), 1e-5f));
    }

    private static Map<String, String> meta() {
        Map<String, String> meta = new LinkedHashMap<>();
        meta.put("seqLen", "16");
        meta.put("steps", "7");
        return meta;
    }

    private static float[] logits(NDManager manager, BlockStackLM model) {
        NDArray mask = CausalMaskGenerator.createCausalMask(manager, 4).squeeze(0).squeeze(0);
        return model.forwardIds(manager, manager.create(new int[]{1, 2, 3, 4}), mask).toFloatArray();
    }

    @Test
    @DisplayName("Architecture, vocabulary, metadata and weights all survive a round trip")
    void testRoundTrip(@TempDir Path dir) throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM original = model(manager);
            Path file = dir.resolve("model.ckpt");
            ModelCheckpoint.save(file, original, TOKENIZER.chars(), meta());

            ModelCheckpoint.Loaded loaded = ModelCheckpoint.load(manager, file);

            assertEquals(original.getConfig(), loaded.model().getConfig());
            assertArrayEquals(TOKENIZER.chars(), loaded.vocabulary());
            assertEquals(meta(), loaded.metadata());
            assertArrayEquals(logits(manager, original), logits(manager, loaded.model()), 1e-6f);
            assertEquals(TOKENIZER.vocabSize(), CharTokenizer.fromChars(loaded.vocabulary()).vocabSize());
        }
    }

    @Test
    @DisplayName("save() leaves no temporary file behind and can overwrite an existing checkpoint")
    void testNoTemporaryFileAndOverwrite(@TempDir Path dir) throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Path file = dir.resolve("model.ckpt");
            ModelCheckpoint.save(file, model(manager), TOKENIZER.chars(), meta());
            ModelCheckpoint.save(file, model(manager), TOKENIZER.chars(), new LinkedHashMap<>());

            assertFalse(Files.exists(dir.resolve("model.ckpt.tmp")));
            assertTrue(ModelCheckpoint.load(manager, file).metadata().isEmpty());
        }
    }

    @Test
    @DisplayName("isCheckpoint() tells the self-describing format from a bare weights file and from a missing file")
    void testIsCheckpoint(@TempDir Path dir) throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = model(manager);
            Path good = dir.resolve("good.ckpt");
            ModelCheckpoint.save(good, model, TOKENIZER.chars(), meta());
            Path legacy = dir.resolve("legacy.bin");
            ModelSerializer.saveWeights(legacy.toString(), model.params());

            assertTrue(ModelCheckpoint.isCheckpoint(good));
            assertFalse(ModelCheckpoint.isCheckpoint(legacy));
            assertFalse(ModelCheckpoint.isCheckpoint(dir.resolve("missing.ckpt")));
        }
    }

    @Test
    @DisplayName("Loading a bare weights file explains what is wrong")
    void testLegacyFileIsExplained(@TempDir Path dir) throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Path legacy = dir.resolve("legacy.bin");
            ModelSerializer.saveWeights(legacy.toString(), model(manager).params());
            IOException e = assertThrows(IOException.class, () -> ModelCheckpoint.load(manager, legacy));
            assertTrue(e.getMessage().contains("self-describing"), e.getMessage());
        }
    }

    @Test
    @DisplayName("A truncated file is rejected instead of producing a half-initialised model")
    void testTruncatedFileIsRejected(@TempDir Path dir) throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Path file = dir.resolve("model.ckpt");
            ModelCheckpoint.save(file, model(manager), TOKENIZER.chars(), meta());
            byte[] all = Files.readAllBytes(file);

            Path cut = dir.resolve("cut.ckpt");
            Files.write(cut, Arrays.copyOf(all, all.length / 2));
            assertThrows(IOException.class, () -> ModelCheckpoint.load(manager, cut));
        }
    }

    @Test
    @DisplayName("A vocabulary that does not match the model is refused when saving")
    void testVocabularySizeIsChecked(@TempDir Path dir) {
        try (NDManager manager = NDManager.newBaseManager()) {
            assertThrows(IllegalArgumentException.class,
                    () -> ModelCheckpoint.save(dir.resolve("x.ckpt"), model(manager), new char[]{'a', 'b'}, meta()));
        }
    }
}
