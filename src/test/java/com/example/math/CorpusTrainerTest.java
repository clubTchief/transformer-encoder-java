package com.example.math;

import ai.djl.ndarray.NDManager;
import com.dictionary.CharCorpus;
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
        CharCorpus corpus = CharCorpus.fromText(story.repeat(200), 1, 0.1);
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

            CharTokenizer tokenizer = CharTokenizer.fromChars(loaded.vocabulary());
            String text = CorpusGeneratorCLI.complete(manager, loaded, tokenizer, 16, "the cat", 40, 0.0f, 0, 1.0f, 1, piece -> { });
            assertTrue(text.startsWith("the cat"));
            assertTrue(text.length() > "the cat".length(), "something was generated");
        }
    }

    @Test
    @DisplayName("Generation can run far past the training window thanks to the sliding context")
    void testGenerationLongerThanTheWindow(@TempDir Path dir) throws Exception {
        CharCorpus corpus = CharCorpus.fromText("abcabcabcabc ".repeat(300), 1, 0.1);
        CorpusTrainer.Settings settings = tinySettings(dir.resolve("abc.ckpt"));
        settings.steps = 40;
        settings.evalEvery = 40;
        CorpusTrainer.train(corpus, settings, new PrintStream(OutputStream.nullOutputStream()));

        try (NDManager manager = NDManager.newBaseManager()) {
            ModelCheckpoint.Loaded loaded = ModelCheckpoint.load(manager, settings.checkpoint);
            CharTokenizer tokenizer = CharTokenizer.fromChars(loaded.vocabulary());
            StringBuilder streamed = new StringBuilder();
            String text = CorpusGeneratorCLI.complete(manager, loaded, tokenizer, 16, "abc", 100, 0.0f, 0, 1.0f, 1, streamed::append);
            assertEquals(text, streamed.toString(), "the sink receives the prompt and every generated piece, in order");
            assertTrue(text.length() >= 3 + 50, "100 tokens requested with a window of 16: " + text.length());
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
}
