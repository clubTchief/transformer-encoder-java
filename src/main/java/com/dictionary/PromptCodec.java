package com.dictionary;

import java.util.ArrayList;
import java.util.List;

/** Turns what a user types into token ids the model was trained on, and generated ids back into text. */
public final class PromptCodec {

    /** @param ids BOS followed by the known words; @param unknownWords words that are not in the vocabulary */
    public record Encoded(int[] ids, List<String> unknownWords) {
    }

    private PromptCodec() {
    }

    /**
     * Same tokenizer as in training (punctuation is split off), BOS first like every training sequence.
     * The vocabulary is case sensitive ("Once"), so a word that is not found as typed is also tried in lower case
     * and capitalised before it is reported as unknown.
     */
    public static Encoded encode(Vocabulary vocab, String input) {
        List<Integer> ids = new ArrayList<>();
        ids.add(vocab.getId(Vocabulary.BOS_TOKEN));
        List<String> unknown = new ArrayList<>();

        if (!input.isBlank()) {
            for (String token : Vocabulary.tokenize(input)) {
                int id = lookup(vocab, token);
                if (id == vocab.getId(Vocabulary.UNK_TOKEN)) {
                    unknown.add(token);
                } else {
                    ids.add(id);
                }
            }
        }
        return new Encoded(ids.stream().mapToInt(Integer::intValue).toArray(), unknown);
    }

    private static int lookup(Vocabulary vocab, String token) {
        int unk = vocab.getId(Vocabulary.UNK_TOKEN);
        int id = vocab.getId(token);
        if (id != unk) {
            return id;
        }
        String lower = token.toLowerCase();
        String capitalised = lower.isEmpty() ? lower : Character.toUpperCase(lower.charAt(0)) + lower.substring(1);
        for (String candidate : new String[]{lower, capitalised}) {
            id = vocab.getId(candidate);
            if (id != unk) {
                return id;
            }
        }
        return unk;
    }

    /** Words up to the first end-of-sequence or padding token; BOS is skipped. */
    public static String decode(Vocabulary vocab, List<Integer> ids) {
        StringBuilder text = new StringBuilder();
        for (int id : ids) {
            String word = vocab.getToken(id);
            if (Vocabulary.EOS_TOKEN.equals(word) || Vocabulary.PAD_TOKEN.equals(word)) {
                break;
            }
            if (Vocabulary.BOS_TOKEN.equals(word)) {
                continue;
            }
            if (text.length() > 0) {
                text.append(' ');
            }
            text.append(word);
        }
        return text.toString();
    }
}
