package com.tokenizer;

/**
 * Turns text into token ids and back. Two implementations exist: a character tokenizer (every character is a token,
 * rare ones become the unknown token) and a byte-level BPE tokenizer (frequent character sequences are single tokens,
 * every possible text can be encoded).
 */
public interface Tokenizer {

    /** Separates documents in a training text; it is encoded as the end-of-text token. */
    String END_OF_TEXT = "<|endoftext|>";

    /** "char" or "bpe": stored in a checkpoint so the matching tokenizer can be rebuilt. */
    String kind();

    int vocabSize();

    /** Id of the end-of-text token. A model that emits it is saying the document is finished. */
    int eosId();

    /** Id of the unknown token, or -1 if every text can be encoded (byte-level BPE). */
    int unkId();

    int[] encode(String text);

    /** The text for these ids; the end-of-text token is written as {@link #END_OF_TEXT}. */
    String decode(int[] ids);

    /** Everything needed to rebuild this tokenizer with the factory for its {@link #kind()}. */
    byte[] serialize();

    /**
     * For streaming generated tokens to a screen. A single BPE token can be half of a UTF-8 character, so tokens
     * cannot be turned into text one by one; this decoder holds back incomplete bytes until the character is whole.
     */
    StreamDecoder newStreamDecoder();

    interface StreamDecoder {

        /** The text that became complete with this token (possibly empty). The end-of-text token produces no text. */
        String accept(int id);

        /** Whatever is still held back, at the end of the stream. */
        String finish();
    }
}
