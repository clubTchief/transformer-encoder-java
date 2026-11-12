package com.example.math;

import ai.djl.ndarray.NDManager;
import com.dictionary.TokenCorpus;
import com.dictionary.TokenizerFactory;
import com.tokenizer.Tokenizer;
import com.dictionary.CorpusGeneratorCLI;
import com.dictionary.CorpusTrainer;
import com.example.story.CharTokenizer;
import com.transformer.ModelCheckpoint;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/** The whole pipeline on a tiny text: train, write the checkpoint, load it again and generate from it. */
class CorpusTrainerTest {

    private static CorpusTrainer.Settings tinySettings(Path checkpoint) {
        CorpusTrainer.Settings s = new CorpusTrainer.Settings();
        s.checkpoint = checkpoint;
        s.numLayers = 1;
        s.dModel = 32;
        s.numHeads = 2;
        s.dFF = 64;
        s.seqLen = 16;
        s.batchSize = 8;
        s.steps = 80;
        s.warmupSteps = 5;
        s.evalEvery = 40;
        s.evalBatches = 2;
        s.sampleTokens = 20;
        s.learningRate = 5e-3f;
        s.seed = 7;
        return s;
    }

    @Test
    @DisplayName("Training lowers the loss, the checkpoint can be loaded again and generates text that continues the prompt")
    void testTrainSaveLoadGenerate(@TempDir Path dir) throws Exception {
        String story = "the cat sat on the mat. the dog sat on the log.<|endoftext|>";
        String text = story.repeat(200);
        TokenCorpus corpus = TokenCorpus.fromText(text, CharTokenizer.fromText(text, 1), 0.1);
        CorpusTrainer.Settings settings = tinySettings(dir.resolve("tiny.ckpt"));

        CorpusTrainer.Result result = CorpusTrainer.train(corpus, settings, new PrintStream(OutputStream.nullOutputStream()));

        assertTrue(result.lastLoss() < result.firstLoss() * 0.8f,
                "loss should fall on such a repetitive text: first " + result.firstLoss() + ", last " + result.lastLoss());
        assertTrue(Float.isFinite(result.validationLoss()));
        assertTrue(ModelCheckpoint.isCheckpoint(settings.checkpoint));

        try (NDManager manager = NDManager.newBaseManager()) {
            ModelCheckpoint.Loaded loaded = ModelCheckpoint.load(manager, settings.checkpoint);
            assertEquals("16", loaded.metadata().get("seqLen"));
            assertEquals("80", loaded.metadata().get("steps"));
            assertEquals(corpus.tokenizer().vocabSize(), loaded.model().getConfig().vocabSize());

            Tokenizer tokenizer = TokenizerFactory.load(loaded);
            String generated = CorpusGeneratorCLI.complete(manager, loaded, tokenizer, 16, "the cat", 40, 0.0f, 0, 1.0f, 1, piece -> { });
            assertTrue(generated.startsWith("the cat"));
            assertTrue(generated.length() > "the cat".length(), "something was generated");
        }
    }

    @Test
    @DisplayName("Generation can run far past the training window thanks to the sliding context")
    void testGenerationLongerThanTheWindow(@TempDir Path dir) throws Exception {
        String text = "abcabcabcabc ".repeat(300);
        TokenCorpus corpus = TokenCorpus.fromText(text, CharTokenizer.fromText(text, 1), 0.1);
        CorpusTrainer.Settings settings = tinySettings(dir.resolve("abc.ckpt"));
        settings.steps = 40;
        settings.evalEvery = 40;
        CorpusTrainer.train(corpus, settings, new PrintStream(OutputStream.nullOutputStream()));

        try (NDManager manager = NDManager.newBaseManager()) {
            ModelCheckpoint.Loaded loaded = ModelCheckpoint.load(manager, settings.checkpoint);
            Tokenizer tokenizer = TokenizerFactory.load(loaded);
            StringBuilder streamed = new StringBuilder();
            String generated = CorpusGeneratorCLI.complete(manager, loaded, tokenizer, 16, "abc", 100, 0.0f, 0, 1.0f, 1, streamed::append);
            assertEquals(generated, streamed.toString(), "the sink receives the prompt and every generated piece, in order");
            assertTrue(generated.length() >= 3 + 50, "100 tokens requested with a window of 16: " + generated.length());
        }
    }

    @Test
    @DisplayName("A misspelled argument is an error, not silently ignored")
    void testUnknownArgumentIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> CorpusTrainer.Settings.fromArgs(new String[]{"corpus=x.txt", "stpes=10"}));
        assertThrows(IllegalArgumentException.class, () -> CorpusTrainer.Settings.fromArgs(new String[]{"corpus"}));
        CorpusTrainer.Settings s = CorpusTrainer.Settings.fromArgs(new String[]{"corpus=x.txt", "steps=10", "lr=0.01", "out=m.ckpt"});
        assertEquals(10, s.steps);
        assertEquals(0.01f, s.learningRate, 1e-9f);
        assertEquals(Path.of("m.ckpt"), s.checkpoint);

        CorpusTrainer.Settings t = CorpusTrainer.Settings.fromArgs(new String[]{"corpus=x.txt", "init=0.05", "logEvery=10"});
        assertEquals(0.05f, t.initStd, 1e-9f);
        assertEquals(10, t.logEvery);
        assertEquals(0.02f, CorpusTrainer.Settings.fromArgs(new String[]{"corpus=x.txt"}).initStd, 1e-9f, "GPT-style init is the default");

        CorpusTrainer.Settings u = CorpusTrainer.Settings.fromArgs(new String[]{"corpus=x.txt", "tokenizer=char", "bpeVocab=1024", "tokenizerFile=t.tok"});
        assertEquals("char", u.tokenizer);
        assertEquals(1024, u.bpeVocab);
        assertEquals(Path.of("t.tok"), u.tokenizerFile);
        assertEquals("bpe", CorpusTrainer.Settings.fromArgs(new String[]{"corpus=x.txt"}).tokenizer, "BPE is the default");
    }

    @Test
    @DisplayName("A missing corpus file produces a message that says where it looked and how to get one")
    void testMissingCorpusFileIsExplained(@TempDir Path dir) {
        CorpusTrainer.Settings settings = tinySettings(dir.resolve("never.ckpt"));
        settings.corpus = dir.resolve("no-such-file.txt");

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> CorpusTrainer.train(settings, new PrintStream(OutputStream.nullOutputStream())));
        assertTrue(e.getMessage().contains("no-such-file.txt"), e.getMessage());
        assertTrue(e.getMessage().contains("curl"), "the message should say how to download TinyStories: " + e.getMessage());

        settings.corpus = null;
        assertThrows(IllegalArgumentException.class,
                () -> CorpusTrainer.train(settings, new PrintStream(OutputStream.nullOutputStream())));
    }

    @Test
    @DisplayName("BPE: the trainer learns the tokenizer, stores it in the checkpoint, and the generator uses it")
    void testTrainWithBpeFromAFile(@TempDir Path dir) throws Exception {
        String story = "the cat sat on the mat. the dog sat on the log. ";
        Path corpusFile = dir.resolve("stories.txt");
        java.nio.file.Files.writeString(corpusFile, story.repeat(300));

        CorpusTrainer.Settings settings = tinySettings(dir.resolve("bpe.ckpt"));
        settings.corpus = corpusFile;
        settings.tokenizer = "bpe";
        settings.bpeVocab = 300;
        settings.tokenizerFile = dir.resolve("bpe.tok");
        settings.steps = 60;
        settings.evalEvery = 60;

        CorpusTrainer.Result result = CorpusTrainer.train(settings, new PrintStream(OutputStream.nullOutputStream()));
        assertTrue(result.lastLoss() < result.firstLoss() * 0.85f,
                "first " + result.firstLoss() + ", last " + result.lastLoss());
        assertTrue(java.nio.file.Files.exists(settings.tokenizerFile), "the learned tokenizer is saved for reuse");

        try (NDManager manager = NDManager.newBaseManager()) {
            ModelCheckpoint.Loaded loaded = ModelCheckpoint.load(manager, settings.checkpoint);
            assertEquals("bpe", loaded.tokenizerKind());
            assertEquals("bpe", loaded.metadata().get("tokenizer"));

            Tokenizer tokenizer = TokenizerFactory.load(loaded);
            assertEquals(tokenizer.vocabSize(), loaded.model().getConfig().vocabSize());
            assertEquals(tokenizer.vocabSize(), TokenizerFactory.load(settings.tokenizerFile).vocabSize());

            StringBuilder streamed = new StringBuilder();
            String generated = CorpusGeneratorCLI.complete(manager, loaded, tokenizer, 16, "the cat", 25, 0.0f, 0, 1.0f, 1, streamed::append);
            assertTrue(generated.startsWith("the cat"));
            assertEquals(generated, streamed.toString());
        }
    }

    @Test
    @DisplayName("An existing tokenizer file is reused instead of learning a new tokenizer")
    void testTokenizerFileIsReused(@TempDir Path dir) throws Exception {
        String text = "the cat sat on the mat. the dog sat on the log. ".repeat(300);
        Path corpusFile = dir.resolve("stories.txt");
        java.nio.file.Files.writeString(corpusFile, text);
        Path tokenizerFile = dir.resolve("saved.tok");
        TokenizerFactory.save(tokenizerFile, com.tokenizer.BpeTrainer.train(text, 280, null)); // smaller than bpeVocab below

        CorpusTrainer.Settings settings = tinySettings(dir.resolve("reuse.ckpt"));
        settings.corpus = corpusFile;
        settings.tokenizer = "bpe";
        settings.bpeVocab = 2048;
        settings.tokenizerFile = tokenizerFile;
        settings.steps = 5;
        settings.evalEvery = 5;
        CorpusTrainer.train(settings, new PrintStream(OutputStream.nullOutputStream()));

        try (NDManager manager = NDManager.newBaseManager()) {
            Tokenizer fromCheckpoint = TokenizerFactory.load(ModelCheckpoint.load(manager, settings.checkpoint));
            assertEquals(TokenizerFactory.load(tokenizerFile).vocabSize(), fromCheckpoint.vocabSize());
        }
    }

    @Test
    @DisplayName("An unknown tokenizer name is refused")
    void testUnknownTokenizerName(@TempDir Path dir) throws Exception {
        Path corpusFile = dir.resolve("stories.txt");
        java.nio.file.Files.writeString(corpusFile, "hello world. ".repeat(100));
        CorpusTrainer.Settings settings = tinySettings(dir.resolve("x.ckpt"));
        settings.corpus = corpusFile;
        settings.tokenizer = "wordpiece";
        assertThrows(IllegalArgumentException.class,
                () -> CorpusTrainer.train(settings, new PrintStream(OutputStream.nullOutputStream())));
    }
}
