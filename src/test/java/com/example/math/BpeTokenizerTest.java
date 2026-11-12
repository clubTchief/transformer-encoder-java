package com.example.math;

import com.tokenizer.BpeTokenizer;
import com.tokenizer.BpeTrainer;
import com.tokenizer.Tokenizer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class BpeTokenizerTest {

    private static final String STORY =
            "Once upon a time, there was a little girl named Lily. She loved to play with her cat. ".repeat(40);

    private static BpeTokenizer trained(int vocabSize) {
        return BpeTrainer.train(STORY, vocabSize, null);
    }

    @Test
    @DisplayName("decode(encode(text)) gives the text back, for any text: accents, emoji, other scripts, whitespace")
    void testRoundTrip() {
        BpeTokenizer tokenizer = trained(400);
        String[] texts = {STORY, "", "x", "h\u00e9llo w\u00f6rld", "emoji \uD83D\uDE00 and \u65E5\u672C\u8A9E",
                "tabs\t and\nnewlines\r\n  spaces   ", "<|endoftext|>", "a<|endoftext|>b", "  leading and trailing  "};
        for (String text : texts) {
            assertEquals(text, tokenizer.decode(tokenizer.encode(text)), "round trip of: " + text);
        }
    }

    @Test
    @DisplayName("Frequent words become single tokens")
    void testFrequentWordsAreSingleTokens() {
        BpeTokenizer tokenizer = trained(600);
        assertEquals(1, tokenizer.encode(" little").length);
        assertEquals(1, tokenizer.encode(" cat").length);
        assertEquals(1, tokenizer.encode(" upon").length);
        assertTrue(tokenizer.encode(STORY).length < STORY.length() / 3, "the text is compressed to well under a third");
    }

    @Test
    @DisplayName("The first merge is the most frequent pair")
    void testFirstMerge() {
        BpeTokenizer tokenizer = BpeTrainer.train("aaabdaaabac", 258, null); // exactly one merge
        assertEquals(258, tokenizer.vocabSize());
        assertEquals("aa", tokenizer.decode(new int[]{257})); // (a, a) occurs 4 times, more than any other pair
        assertEquals(1, tokenizer.encode("aa").length);
    }

    @Test
    @DisplayName("Training stops when no pair repeats, and a rare word stays in pieces")
    void testStopsWhenNothingRepeats() {
        BpeTokenizer tokenizer = BpeTrainer.train("abcdefg", 400, null);
        assertEquals(257, tokenizer.vocabSize(), "no merges were learned");
        assertEquals(7, tokenizer.encode("abcdefg").length);
    }

    @Test
    @DisplayName("The same text always gives the same tokenizer")
    void testTrainingIsDeterministic() {
        assertArrayEquals(trained(500).serialize(), trained(500).serialize());
    }

    @Test
    @DisplayName("A serialized tokenizer encodes exactly like the original")
    void testSerializationRoundTrip() {
        BpeTokenizer original = trained(500);
        BpeTokenizer copy = BpeTokenizer.deserialize(original.serialize());
        assertEquals(original.vocabSize(), copy.vocabSize());
        assertArrayEquals(original.encode(STORY), copy.encode(STORY));
        assertArrayEquals(original.encode("A new sentence about a cat."), copy.encode("A new sentence about a cat."));
    }

    @Test
    @DisplayName("Damaged tokenizer data is rejected")
    void testDamagedData() {
        assertThrows(IllegalArgumentException.class, () -> BpeTokenizer.deserialize(new byte[]{1, 2, 3}));
        byte[] data = trained(300).serialize();
        assertThrows(IllegalArgumentException.class, () -> BpeTokenizer.deserialize(Arrays.copyOf(data, data.length - 5)));
    }

    @Test
    @DisplayName("End of text is token 0 and the markers inside a text become that token")
    void testEndOfText() {
        BpeTokenizer tokenizer = trained(400);
        assertEquals(0, tokenizer.eosId());
        int[] ids = tokenizer.encode("a<|endoftext|>b");
        assertEquals(3, ids.length);
        assertEquals(0, ids[1]);
        assertEquals("a", tokenizer.decode(new int[]{ids[0]}));
        assertEquals("b", tokenizer.decode(new int[]{ids[2]}));
    }

    @Test
    @DisplayName("Byte-level BPE has no unknown token")
    void testNoUnknownToken() {
        assertEquals(-1, trained(300).unkId());
        assertEquals("bpe", trained(300).kind());
    }

    @Test
    @DisplayName("Merges never cross a word boundary: no learned token has a space after a letter")
    void testMergesStayInsideWords() {
        BpeTokenizer tokenizer = trained(600);
        for (int id = 257; id < tokenizer.vocabSize(); id++) {
            String token = tokenizer.decode(new int[]{id});
            assertFalse(token.matches("(?s).*\\p{L} .*"), "token '" + token + "' joins two words");
        }
    }

    @Test
    @DisplayName("The stream decoder never emits half a character")
    void testStreamDecoderWaitsForCompleteCharacters() {
        BpeTokenizer tokenizer = trained(300); // never saw these characters: every one is a few single-byte tokens
        String text = "caf\u00e9 \uD83D\uDE00 \u65E5\u672C\u8A9E";
        int[] ids = tokenizer.encode(text);

        Tokenizer.StreamDecoder decoder = tokenizer.newStreamDecoder();
        StringBuilder streamed = new StringBuilder();
        for (int id : ids) {
            String piece = decoder.accept(id);
            assertFalse(piece.contains("\uFFFD"), "a broken character was emitted: " + piece);
            streamed.append(piece);
        }
        streamed.append(decoder.finish());
        assertEquals(text, streamed.toString());
    }

    @Test
    @DisplayName("The stream decoder produces no text for the end-of-text token")
    void testStreamDecoderSkipsEndOfText() {
        BpeTokenizer tokenizer = trained(300);
        Tokenizer.StreamDecoder decoder = tokenizer.newStreamDecoder();
        assertEquals("", decoder.accept(tokenizer.eosId()));
        assertEquals("", decoder.finish());
    }

    @Test
    @DisplayName("A vocabulary size that leaves no room for merges is rejected")
    void testVocabularyTooSmall() {
        assertThrows(IllegalArgumentException.class, () -> BpeTrainer.train(STORY, 257, null));
    }

    @Test
    @DisplayName("Token ids outside the vocabulary are rejected when decoding")
    void testDecodeRejectsUnknownIds() {
        BpeTokenizer tokenizer = trained(300);
        assertThrows(IllegalArgumentException.class, () -> tokenizer.decode(new int[]{tokenizer.vocabSize()}));
        assertThrows(IllegalArgumentException.class, () -> tokenizer.decode(new int[]{-1}));
    }

    @Test
    @DisplayName("The trainer reports progress, at the latest when training ends")
    void testProgressIsReported() {
        int[] calls = {0};
        int[] lastDone = {-1};
        BpeTokenizer tokenizer = BpeTrainer.train(STORY, 500, (done, total, count) -> {
            calls[0]++;
            lastDone[0] = done;
        });
        assertTrue(calls[0] >= 1);
        assertEquals(tokenizer.vocabSize() - 257, lastDone[0], "the last report is the final number of merges");
    }
}
