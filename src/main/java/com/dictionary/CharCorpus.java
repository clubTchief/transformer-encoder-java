package com.dictionary;

import com.example.story.CharTokenizer;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

/**
 * A text file as one long stream of character ids (stories are separated by the end-of-text id), split into a
 * training part and a validation part. Training batches are random windows cut from anywhere in the training part,
 * so every position of the text is eventually seen with every kind of left context. Unlike the fixed, non-overlapping
 * chunks of StoryDataset this never cuts a sentence off from its beginning.
 */
public final class CharCorpus {

    private final CharTokenizer tokenizer;
    private final int[] ids;
    private final int split; // ids[0, split) = training, ids[split, ids.length) = validation

    private CharCorpus(CharTokenizer tokenizer, int[] ids, int split) {
        this.tokenizer = tokenizer;
        this.ids = ids;
        this.split = split;
    }

    /**
     * @param maxChars           read at most this many characters from the start of the file
     * @param minCharFreq        characters rarer than this become the unknown token
     * @param validationFraction share of the text (taken from the END) kept for validation, 0 to 0.5
     */
    public static CharCorpus load(Path file, int maxChars, int minCharFreq, double validationFraction) throws IOException {
        StringBuilder text = new StringBuilder();
        try (BufferedReader reader = Files.newBufferedReader(file)) {
            char[] buffer = new char[1 << 16];
            int read;
            while (text.length() < maxChars && (read = reader.read(buffer)) > 0) {
                text.append(buffer, 0, read);
            }
        }
        return fromText(text.toString(), minCharFreq, validationFraction);
    }

    public static CharCorpus fromText(String text, int minCharFreq, double validationFraction) {
        if (text.isEmpty()) {
            throw new IllegalArgumentException("The corpus is empty");
        }
        if (validationFraction < 0 || validationFraction > 0.5) {
            throw new IllegalArgumentException("validationFraction must be between 0 and 0.5 but was " + validationFraction);
        }
        CharTokenizer tokenizer = CharTokenizer.fromText(text, minCharFreq);
        int[] ids = tokenizer.encode(text);
        int validation = (int) Math.round(ids.length * validationFraction);
        return new CharCorpus(tokenizer, ids, ids.length - validation);
    }

    public CharTokenizer tokenizer() {
        return tokenizer;
    }

    public int trainTokens() {
        return split;
    }

    public int validationTokens() {
        return ids.length - split;
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
