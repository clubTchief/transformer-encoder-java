package com.dictionary;

/**
 *
 * @author admin
 */
import java.util.*;

public class Vocabulary {
    public static final String PAD_TOKEN = "<PAD>";
    public static final String UNK_TOKEN = "<UNK>";
    public static final String BOS_TOKEN = "<BOS>";
    public static final String EOS_TOKEN = "<EOS>";

    private final Map<String, Integer> tokenToId = new HashMap<>();
    private final Map<Integer, String> idToToken = new HashMap<>();

    public Vocabulary(List<String> corpus) {
        // Reserve special tokens
        addToken(PAD_TOKEN);
        addToken(UNK_TOKEN);
        addToken(BOS_TOKEN);
        addToken(EOS_TOKEN);

        // Populate vocabulary from corpus
        for (String sentence : corpus) {
            String[] tokens = tokenize(sentence);
            for (String token : tokens) {
                addToken(token);
            }
        }
    }

    private void addToken(String token) {
        if (!tokenToId.containsKey(token)) {
            int id = tokenToId.size();
            tokenToId.put(token, id);
            idToToken.put(id, token);
        }
    }

    public static String[] tokenize(String text) {
        // Basic word tokenizer handling punctuation separation
        return text.replaceAll("([.,!?;:])", " $1 ")
                   .trim()
                   .split("\\s+");
    }

    public int getId(String token) {
        return tokenToId.getOrDefault(token, tokenToId.get(UNK_TOKEN));
    }

    public String getToken(int id) {
        return idToToken.getOrDefault(id, UNK_TOKEN);
    }

    public int getPadId() {
        return tokenToId.get(PAD_TOKEN);
    }

    public int size() {
        return tokenToId.size();
    }
}