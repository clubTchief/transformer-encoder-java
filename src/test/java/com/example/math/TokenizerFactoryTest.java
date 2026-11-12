package com.example.math;

import com.dictionary.TokenizerFactory;
import com.example.story.CharTokenizer;
import com.tokenizer.BpeTrainer;
import com.tokenizer.Tokenizer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class TokenizerFactoryTest {

    private static final String TEXT = "the cat sat on the mat. the dog sat on the log.<|endoftext|>".repeat(30);

    @Test
    @DisplayName("Both kinds are rebuilt from their kind and data")
    void testRebuildByKind() {
        Tokenizer chars = CharTokenizer.fromText(TEXT, 1);
        Tokenizer bpe = BpeTrainer.train(TEXT, 300, null);

        Tokenizer chars2 = TokenizerFactory.load(chars.kind(), chars.serialize());
        Tokenizer bpe2 = TokenizerFactory.load(bpe.kind(), bpe.serialize());

        assertEquals("char", chars2.kind());
        assertEquals("bpe", bpe2.kind());
        assertArrayEquals(chars.encode(TEXT), chars2.encode(TEXT));
        assertArrayEquals(bpe.encode(TEXT), bpe2.encode(TEXT));
        assertEquals(chars.vocabSize(), chars2.vocabSize());
        assertEquals(bpe.vocabSize(), bpe2.vocabSize());
    }

    @Test
    @DisplayName("An unknown kind is refused")
    void testUnknownKind() {
        assertThrows(IllegalArgumentException.class, () -> TokenizerFactory.load("wordpiece", new byte[0]));
    }

    @Test
    @DisplayName("A tokenizer file round-trips, and other files are refused")
    void testTokenizerFile(@TempDir Path dir) throws IOException {
        Tokenizer bpe = BpeTrainer.train(TEXT, 300, null);
        Path file = dir.resolve("nested/bpe.tok");
        TokenizerFactory.save(file, bpe);

        Tokenizer loaded = TokenizerFactory.load(file);
        assertEquals("bpe", loaded.kind());
        assertArrayEquals(bpe.encode(TEXT), loaded.encode(TEXT));

        Path other = dir.resolve("other.txt");
        Files.writeString(other, "this is not a tokenizer file");
        assertThrows(IOException.class, () -> TokenizerFactory.load(other));
    }

    @Test
    @DisplayName("The character tokenizer keeps its behaviour behind the common interface")
    void testCharTokenizerThroughTheInterface() {
        Tokenizer chars = CharTokenizer.fromText(TEXT, 1);
        assertEquals(CharTokenizer.EOS, chars.eosId());
        assertEquals(CharTokenizer.UNK, chars.unkId());
        assertEquals("the cat", chars.decode(chars.encode("the cat")));
        assertEquals("c<|endoftext|>t", chars.decode(chars.encode("c<|endoftext|>t")));

        Tokenizer.StreamDecoder decoder = chars.newStreamDecoder();
        assertEquals("t", decoder.accept(chars.encode("t")[0]));
        assertEquals("", decoder.accept(chars.eosId()));
    }
}
