package com.example.story;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Character-level tokenizer. Chosen on purpose: vocab of ~80-100 keeps the tied embedding/output
 * matrix tiny and needs no tokenizer training. Id 0 = end-of-story, id 1 = unknown.
 * Upgrade path: swap in a small BPE (the HF tokenizers dependency is already in the pom).
 */
public final class CharTokenizer {
    public static final int EOS = 0;
    public static final int UNK = 1;
    private static final char EOS_CHAR = '\u0003';
    private static final String EOS_TEXT = "<|endoftext|>"; // TinyStories story separator

    private final char[] idToChar;
    private final Map<Character, Integer> charToId = new HashMap<>();

    private CharTokenizer(char[] chars) {
        this.idToChar = chars;
        for (int i = 2; i < chars.length; i++) charToId.put(chars[i], i);
    }

    public static CharTokenizer fromText(String text, int minFreq) {
        String t = text.replace(EOS_TEXT, "");
        Map<Character, Integer> freq = new HashMap<>();
        for (int i = 0; i < t.length(); i++) freq.merge(t.charAt(i), 1, Integer::sum);
        List<Character> keep = freq.entrySet().stream()
                .filter(e -> e.getValue() >= minFreq)
                .map(Map.Entry::getKey).sorted().toList();
        char[] chars = new char[keep.size() + 2];
        chars[EOS] = EOS_CHAR;
        chars[UNK] = '?';
        for (int i = 0; i < keep.size(); i++) chars[i + 2] = keep.get(i);
        return new CharTokenizer(chars);
    }

    public int vocabSize() {
        return idToChar.length;
    }

    public int[] encode(String text) {
        String t = text.replace(EOS_TEXT, String.valueOf(EOS_CHAR));
        int[] ids = new int[t.length()];
        for (int i = 0; i < ids.length; i++) {
            char c = t.charAt(i);
            ids[i] = (c == EOS_CHAR) ? EOS : charToId.getOrDefault(c, UNK);
        }
        return ids;
    }

    public String decodeToken(int id) {
        return id == EOS ? "" : String.valueOf(idToChar[id]);
    }

    public void save(Path file) throws IOException {
        StringBuilder sb = new StringBuilder();
        for (char c : idToChar) sb.append((int) c).append('\n');
        Files.writeString(file, sb.toString());
    }

    public static CharTokenizer load(Path file) throws IOException {
        List<String> lines = Files.readAllLines(file);
        char[] chars = new char[lines.size()];
        for (int i = 0; i < chars.length; i++) chars[i] = (char) Integer.parseInt(lines.get(i).trim());
        return new CharTokenizer(chars);
    }
}
