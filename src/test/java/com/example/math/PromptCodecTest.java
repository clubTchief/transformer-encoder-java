package com.example.math;

import com.dictionary.PromptCodec;
import com.dictionary.StoryModelSpec;
import com.dictionary.Vocabulary;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class PromptCodecTest {

    private final Vocabulary vocab = StoryModelSpec.vocabulary();

    private int id(String token) {
        return vocab.getId(token);
    }

    @Test
    @DisplayName("BOS first, then the words; the vocabulary's capitalisation is found from lower case")
    void testEncodesLikeTraining() {
        PromptCodec.Encoded e = PromptCodec.encode(vocab, "once upon a time");
        assertArrayEquals(new int[]{id(Vocabulary.BOS_TOKEN), id("Once"), id("upon"), id("a"), id("time")}, e.ids());
        assertTrue(e.unknownWords().isEmpty());
    }

    @Test
    @DisplayName("Punctuation is split off exactly like in training")
    void testPunctuationIsSplit() {
        PromptCodec.Encoded e = PromptCodec.encode(vocab, "brave knight.");
        assertArrayEquals(new int[]{id(Vocabulary.BOS_TOKEN), id("brave"), id("knight"), id(".")}, e.ids());
    }

    @Test
    @DisplayName("Unknown words are reported and left out, not silently fed to the model as <UNK>")
    void testUnknownWordsAreReported() {
        PromptCodec.Encoded e = PromptCodec.encode(vocab, "banana knight");
        assertArrayEquals(new int[]{id(Vocabulary.BOS_TOKEN), id("knight")}, e.ids());
        assertEquals(List.of("banana"), e.unknownWords());
    }

    @Test
    @DisplayName("An empty prompt is just BOS")
    void testEmptyPrompt() {
        assertArrayEquals(new int[]{id(Vocabulary.BOS_TOKEN)}, PromptCodec.encode(vocab, "   ").ids());
    }

    @Test
    @DisplayName("decode() skips BOS and stops at the first EOS or PAD")
    void testDecodeStopsAtEos() {
        List<Integer> ids = List.of(id(Vocabulary.BOS_TOKEN), id("brave"), id("knight"),
                id(Vocabulary.EOS_TOKEN), id("dragon"));
        assertEquals("brave knight", PromptCodec.decode(vocab, ids));
    }

    @Test
    @DisplayName("The shared spec produces a valid model configuration for the corpus vocabulary")
    void testSpecIsConsistent() {
        assertEquals(StoryModelSpec.D_MODEL / StoryModelSpec.NUM_HEADS, StoryModelSpec.config(vocab.size()).headDim());
        assertEquals(vocab.size(), StoryModelSpec.config(vocab.size()).vocabSize());
    }
}
