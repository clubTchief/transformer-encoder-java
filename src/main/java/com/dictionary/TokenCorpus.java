package com.dictionary;

import com.tokenizer.Tokenizer;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

/**
 * A text as one long stream of token ids (documents are separated by the end-of-text token), split into a training
 * part and a validation part. Training batches are random windows cut from anywhere in the training part, so every
 * position of the text is eventually seen with every kind of left context. Unlike the fixed, non-overlapping chunks of
 * StoryDataset this never cuts a sentence off from its beginning. Works with any {@link Tokenizer}.
 */
public final class TokenCorpus {

    private final Tokenizer tokenizer;
    private final int[] ids;
    private final int split; // ids[0, split) = training, ids[split, ids.length) = validation
    private final long textLength;

    private TokenCorpus(Tokenizer tokenizer, int[] ids, int split, long textLength) {
        this.tokenizer = tokenizer;
        this.ids = ids;
        this.split = split;
        this.textLength = textLength;
    }

    /** Reads at most {@code maxChars} characters from the start of the file. */
    public static String readText(Path file, int maxChars) throws IOException {
        StringBuilder text = new StringBuilder();
        try (BufferedReader reader = Files.newBufferedReader(file)) {
            char[] buffer = new char[1 << 16];
            int read;
            while (text.length() < maxChars && (read = reader.read(buffer)) > 0) {
                text.append(buffer, 0, read);
            }
        }
        return text.toString();
    }

    /**
     * @param validationFraction share of the tokens (taken from the END) kept for validation, 0 to 0.5
     */
    public static TokenCorpus fromText(String text, Tokenizer tokenizer, double validationFraction) {
        if (text.isEmpty()) {
            throw new IllegalArgumentException("The corpus is empty");
        }
        if (validationFraction < 0 || validationFraction > 0.5) {
            throw new IllegalArgumentException("validationFraction must be between 0 and 0.5 but was " + validationFraction);
        }
        int[] ids = tokenizer.encode(text);
        int validation = (int) Math.round(ids.length * validationFraction);
        return new TokenCorpus(tokenizer, ids, ids.length - validation, text.length());
    }

    public Tokenizer tokenizer() {
        return tokenizer;
    }

    public int trainTokens() {
        return split;
    }

    public int validationTokens() {
        return ids.length - split;
    }

    /** Characters of text per token: about 1 for a character tokenizer, 3 to 4 for BPE on English. */
    public double charsPerToken() {
        return ids.length == 0 ? 0 : (double) textLength / ids.length;
    }

    /** Fills inputs[b][t] = ids[o + t] and targets[b][t] = ids[o + t + 1] for random offsets o in the training part. */
    public void sampleTrainBatch(Random rng, int[][] inputs, int[][] targets) {
        fill(0, split, rng, inputs, targets);
    }

    /**
     * Same, from the validation part. Without validation text (or with too little of it for one window) the training
     * part is used instead, so tiny corpora still work; the number is then not a held-out loss.
     */
    public void sampleValidationBatch(Random rng, int[][] inputs, int[][] targets) {
        int window = inputs[0].length;
        if (validationTokens() >= window + 1) {
            fill(split, ids.length, rng, inputs, targets);
        } else {
            fill(0, split, rng, inputs, targets);
        }
    }

    public boolean hasHeldOutValidation(int window) {
        return validationTokens() >= window + 1;
    }

    private void fill(int from, int to, Random rng, int[][] inputs, int[][] targets) {
        int window = inputs[0].length;
        if (to - from < window + 1) {
            throw new IllegalArgumentException("This part of the text has " + (to - from) + " tokens but one window needs "
                    + (window + 1) + ". Use a longer corpus or a shorter seqLen.");
        }
        for (int b = 0; b < inputs.length; b++) {
            int offset = from + rng.nextInt(to - from - window); // offset + window <= to - 1: the last target exists
            for (int t = 0; t < window; t++) {
                inputs[b][t] = ids[offset + t];
                targets[b][t] = ids[offset + t + 1];
            }
        }
    }
}
