package com.example.math;

import com.dictionary.CharCorpus;
import com.example.story.CharTokenizer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class CharCorpusTest {

    @Test
    @DisplayName("Training windows come only from the training part, validation windows only from the validation part")
    void testPartsAreKeptApart() {
        CharCorpus corpus = CharCorpus.fromText("a".repeat(1000) + "b".repeat(1000), 1, 0.5);
        int a = corpus.tokenizer().encode("a")[0];
        int b = corpus.tokenizer().encode("b")[0];
        assertEquals(1000, corpus.trainTokens());
        assertEquals(1000, corpus.validationTokens());

        int[][] x = new int[16][32];
        int[][] y = new int[16][32];
        corpus.sampleTrainBatch(new Random(1), x, y);
        for (int r = 0; r < 16; r++) {
            for (int t = 0; t < 32; t++) {
                assertEquals(a, x[r][t]);
                assertEquals(a, y[r][t]);
            }
        }
        corpus.sampleValidationBatch(new Random(1), x, y);
        for (int r = 0; r < 16; r++) {
            for (int t = 0; t < 32; t++) {
                assertEquals(b, x[r][t]);
                assertEquals(b, y[r][t]);
            }
        }
    }

    @Test
    @DisplayName("Every target is the next character of its input window")
    void testTargetsAreInputsShiftedByOne() {
        CharCorpus corpus = CharCorpus.fromText("abcdefghij".repeat(100), 1, 0.1); // ids 2..11 in alphabetical order
        int[][] x = new int[8][16];
        int[][] y = new int[8][16];
        corpus.sampleTrainBatch(new Random(2), x, y);
        for (int r = 0; r < 8; r++) {
            for (int t = 0; t < 16; t++) {
                assertEquals((x[r][t] - 2 + 1) % 10 + 2, y[r][t], "the next letter of the alphabet pattern");
                if (t < 15) {
                    assertEquals(y[r][t], x[r][t + 1], "targets are the inputs shifted by one");
                }
            }
        }
    }

    @Test
    @DisplayName("The same seed gives the same batch")
    void testSeedIsReproducible() {
        CharCorpus corpus = CharCorpus.fromText("the quick brown fox jumps over the lazy dog ".repeat(50), 1, 0.1);
        int[][] x1 = new int[4][12], y1 = new int[4][12], x2 = new int[4][12], y2 = new int[4][12];
        corpus.sampleTrainBatch(new Random(5), x1, y1);
        corpus.sampleTrainBatch(new Random(5), x2, y2);
        assertArrayEquals(x1, x2);
        assertArrayEquals(y1, y2);
    }

    @Test
    @DisplayName("The end-of-text marker of the corpus becomes the EOS id")
    void testEndOfTextMarker() {
        CharCorpus corpus = CharCorpus.fromText("ab<|endoftext|>ab", 1, 0.0); // 5 tokens: a b EOS a b
        assertEquals(5, corpus.trainTokens());
        int[][] x = new int[1][4];
        int[][] y = new int[1][4];
        corpus.sampleTrainBatch(new Random(1), x, y); // the only possible window starts at 0
        assertEquals(CharTokenizer.EOS, x[0][2]);
        assertEquals(CharTokenizer.EOS, y[0][1]);
    }

    @Test
    @DisplayName("Too little text for one window is an error that says what to do")
    void testTooShortText() {
        CharCorpus corpus = CharCorpus.fromText("abc", 1, 0.0);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> corpus.sampleTrainBatch(new Random(1), new int[1][8], new int[1][8]));
        assertTrue(e.getMessage().contains("seqLen"), e.getMessage());
    }

    @Test
    @DisplayName("Without held-out text the validation batches fall back to the training part")
    void testValidationFallback() {
        CharCorpus corpus = CharCorpus.fromText("abcdefghij".repeat(20), 1, 0.0);
        assertEquals(0, corpus.validationTokens());
        assertFalse(corpus.hasHeldOutValidation(16));
        assertDoesNotThrow(() -> corpus.sampleValidationBatch(new Random(1), new int[2][16], new int[2][16]));
    }

    @Test
    @DisplayName("Empty text and an invalid validation fraction are rejected")
    void testInvalidInput() {
        assertThrows(IllegalArgumentException.class, () -> CharCorpus.fromText("", 1, 0.1));
        assertThrows(IllegalArgumentException.class, () -> CharCorpus.fromText("abc", 1, 0.9));
        assertThrows(IllegalArgumentException.class, () -> CharCorpus.fromText("abc", 1, -0.1));
    }
}
