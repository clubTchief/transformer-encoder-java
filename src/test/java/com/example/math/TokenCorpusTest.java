package com.example.math;

import com.dictionary.TokenCorpus;
import com.example.story.CharTokenizer;
import com.tokenizer.BpeTrainer;
import com.tokenizer.Tokenizer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class TokenCorpusTest {

    private static TokenCorpus chars(String text, double validationFraction) {
        return TokenCorpus.fromText(text, CharTokenizer.fromText(text, 1), validationFraction);
    }

    @Test
    @DisplayName("Training windows come only from the training part, validation windows only from the validation part")
    void testPartsAreKeptApart() {
        TokenCorpus corpus = chars("a".repeat(1000) + "b".repeat(1000), 0.5);
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
    @DisplayName("Every target is the next token of its input window")
    void testTargetsAreInputsShiftedByOne() {
        TokenCorpus corpus = chars("abcdefghij".repeat(100), 0.1); // ids 2..11 in alphabetical order
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
        TokenCorpus corpus = chars("the quick brown fox jumps over the lazy dog ".repeat(50), 0.1);
        int[][] x1 = new int[4][12], y1 = new int[4][12], x2 = new int[4][12], y2 = new int[4][12];
        corpus.sampleTrainBatch(new Random(5), x1, y1);
        corpus.sampleTrainBatch(new Random(5), x2, y2);
        assertArrayEquals(x1, x2);
        assertArrayEquals(y1, y2);
    }

    @Test
    @DisplayName("The end-of-text marker of the corpus becomes the EOS id")
    void testEndOfTextMarker() {
        TokenCorpus corpus = chars("ab<|endoftext|>ab", 0.0); // 5 tokens: a b EOS a b
        assertEquals(5, corpus.trainTokens());
        int[][] x = new int[1][4];
        int[][] y = new int[1][4];
        corpus.sampleTrainBatch(new Random(1), x, y); // the only possible window starts at 0
        assertEquals(corpus.tokenizer().eosId(), x[0][2]);
        assertEquals(corpus.tokenizer().eosId(), y[0][1]);
    }

    @Test
    @DisplayName("Too little text for one window is an error that says what to do")
    void testTooShortText() {
        TokenCorpus corpus = chars("abc", 0.0);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> corpus.sampleTrainBatch(new Random(1), new int[1][8], new int[1][8]));
        assertTrue(e.getMessage().contains("seqLen"), e.getMessage());
    }

    @Test
    @DisplayName("Without held-out text the validation batches fall back to the training part")
    void testValidationFallback() {
        TokenCorpus corpus = chars("abcdefghij".repeat(20), 0.0);
        assertEquals(0, corpus.validationTokens());
        assertFalse(corpus.hasHeldOutValidation(16));
        assertDoesNotThrow(() -> corpus.sampleValidationBatch(new Random(1), new int[2][16], new int[2][16]));
    }

    @Test
    @DisplayName("Empty text and an invalid validation fraction are rejected")
    void testInvalidInput() {
        Tokenizer tokenizer = CharTokenizer.fromText("abc", 1);
        assertThrows(IllegalArgumentException.class, () -> TokenCorpus.fromText("", tokenizer, 0.1));
        assertThrows(IllegalArgumentException.class, () -> TokenCorpus.fromText("abc", tokenizer, 0.9));
        assertThrows(IllegalArgumentException.class, () -> TokenCorpus.fromText("abc", tokenizer, -0.1));
    }

    @Test
    @DisplayName("A BPE corpus has far fewer tokens than characters, and windows still shift by one")
    void testBpeCorpus() {
        String text = "the cat sat on the mat. the dog sat on the log. ".repeat(200);
        Tokenizer bpe = BpeTrainer.train(text, 400, null);
        TokenCorpus corpus = TokenCorpus.fromText(text, bpe, 0.1);

        assertTrue(corpus.charsPerToken() > 3.0, "characters per token: " + corpus.charsPerToken());
        assertEquals(bpe.encode(text).length, corpus.trainTokens() + corpus.validationTokens());

        int[][] x = new int[4][8];
        int[][] y = new int[4][8];
        corpus.sampleTrainBatch(new Random(3), x, y);
        for (int r = 0; r < 4; r++) {
            for (int t = 0; t < 7; t++) {
                assertEquals(y[r][t], x[r][t + 1]);
            }
        }
    }
}
