package com.example.math;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.dictionary.TokenizerFactory;
import com.example.story.CharTokenizer;
import com.tokenizer.BpeTrainer;
import com.tokenizer.Tokenizer;
import com.transformer.BlockStackLM;
import com.transformer.CausalMaskGenerator;
import com.transformer.LMConfig;
import com.transformer.ModelCheckpoint;
import com.transformer.ModelSerializer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
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

    // ------------------------------------------------------------------ tokenizer kinds and the old format

    @Test
    @DisplayName("A BPE tokenizer is stored in the checkpoint and comes back with the same behaviour")
    void testBpeTokenizerRoundTrip(@TempDir Path dir) throws IOException {
        String text = "the cat sat on the mat. the dog sat on the log.<|endoftext|>".repeat(40);
        Tokenizer bpe = BpeTrainer.train(text, 300, null);
        try (NDManager manager = NDManager.newBaseManager()) {
            Engine.getInstance().setRandomSeed(3);
            BlockStackLM model = new BlockStackLM(manager, new LMConfig(2, 16, 2, 32, bpe.vocabSize(), 1e-5f));
            Path file = dir.resolve("bpe.ckpt");
            ModelCheckpoint.save(file, model, bpe, meta());

            ModelCheckpoint.Loaded loaded = ModelCheckpoint.load(manager, file);
            assertEquals("bpe", loaded.tokenizerKind());
            Tokenizer restored = TokenizerFactory.load(loaded);
            assertEquals(bpe.vocabSize(), restored.vocabSize());
            assertArrayEquals(bpe.encode(text), restored.encode(text));
            assertThrows(IllegalStateException.class, loaded::vocabulary, "a BPE checkpoint has no character vocabulary");
        }
    }

    @Test
    @DisplayName("A tokenizer whose size does not match the model is refused when saving")
    void testTokenizerSizeIsChecked(@TempDir Path dir) {
        Tokenizer bpe = BpeTrainer.train("the cat sat on the mat. ".repeat(50), 300, null);
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM model = new BlockStackLM(manager, new LMConfig(1, 16, 2, 32, bpe.vocabSize() + 1, 1e-5f));
            assertThrows(IllegalArgumentException.class, () -> ModelCheckpoint.save(dir.resolve("x.ckpt"), model, bpe, meta()));
        }
    }

    /** Writes the format of version 1, which stored a bare character vocabulary: what the first corpus trainer produced. */
    private static void writeVersion1(Path file, BlockStackLM model, char[] vocabulary, Map<String, String> meta) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(file)))) {
            LMConfig c = model.getConfig();
            out.writeInt(ModelCheckpoint.MAGIC);
            out.writeInt(1);
            out.writeInt(c.numLayers());
            out.writeInt(c.dModel());
            out.writeInt(c.numHeads());
            out.writeInt(c.dFF());
            out.writeInt(c.vocabSize());
            out.writeFloat(c.eps());
            out.writeInt(vocabulary.length);
            for (char ch : vocabulary) out.writeChar(ch);
            out.writeInt(meta.size());
            for (Map.Entry<String, String> e : meta.entrySet()) {
                out.writeUTF(e.getKey());
                out.writeUTF(e.getValue());
            }
            ModelSerializer.writeTensors(out, model.params());
        }
    }

    @Test
    @DisplayName("Checkpoints written before BPE existed (version 1) still load, as character-tokenizer checkpoints")
    void testVersion1CheckpointsStillLoad(@TempDir Path dir) throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            BlockStackLM original = model(manager);
            Path file = dir.resolve("old.ckpt");
            writeVersion1(file, original, TOKENIZER.chars(), meta());

            assertTrue(ModelCheckpoint.isCheckpoint(file));
            ModelCheckpoint.Loaded loaded = ModelCheckpoint.load(manager, file);

            assertEquals("char", loaded.tokenizerKind());
            assertArrayEquals(TOKENIZER.chars(), loaded.vocabulary());
            assertEquals(meta(), loaded.metadata());
            assertArrayEquals(logits(manager, original), logits(manager, loaded.model()), 1e-6f);

            Tokenizer restored = TokenizerFactory.load(loaded);
            assertArrayEquals(TOKENIZER.encode("abc xyz"), restored.encode("abc xyz"));
        }
    }

    @Test
    @DisplayName("A future checkpoint version is refused with a clear message")
    void testFutureVersionIsRefused(@TempDir Path dir) throws IOException {
        Path file = dir.resolve("future.ckpt");
        try (DataOutputStream out = new DataOutputStream(Files.newOutputStream(file))) {
            out.writeInt(ModelCheckpoint.MAGIC);
            out.writeInt(99);
        }
        try (NDManager manager = NDManager.newBaseManager()) {
            IOException e = assertThrows(IOException.class, () -> ModelCheckpoint.load(manager, file));
            assertTrue(e.getMessage().contains("version"), e.getMessage());
        }
    }
}
