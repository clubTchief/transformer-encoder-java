package com.example.story;

import com.tokenizer.Tokenizer;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
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
public final class CharTokenizer implements Tokenizer {
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

    @Override
    public String kind() {
        return "char";
    }

    @Override
    public int eosId() {
        return EOS;
    }

    @Override
    public int unkId() {
        return UNK;
    }

    @Override
    public String decode(int[] ids) {
        StringBuilder text = new StringBuilder();
        for (int id : ids) {
            text.append(id == EOS ? END_OF_TEXT : decodeToken(id));
        }
        return text.toString();
    }

    /** The vocabulary as UTF-16 characters, two bytes each. */
    @Override
    public byte[] serialize() {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        try (DataOutputStream out = new DataOutputStream(buffer)) {
            for (char c : idToChar) {
                out.writeChar(c);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return buffer.toByteArray();
    }

    public static CharTokenizer deserialize(byte[] data) {
        if (data.length % 2 != 0) {
            throw new IllegalArgumentException("Damaged character vocabulary: odd number of bytes");
        }
        char[] chars = new char[data.length / 2];
        try (DataInputStream in = new DataInputStream(new ByteArrayInputStream(data))) {
            for (int i = 0; i < chars.length; i++) {
                chars[i] = in.readChar();
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return fromChars(chars);
    }

    @Override
    public StreamDecoder newStreamDecoder() {
        return new StreamDecoder() {
            @Override
            public String accept(int id) {
                return decodeToken(id);
            }

            @Override
            public String finish() {
                return "";
            }
        };
    }

    /** The vocabulary, id = position (a copy). Store it in a checkpoint to rebuild this tokenizer later. */
    public char[] chars() {
        return idToChar.clone();
    }

    /** Rebuilds a tokenizer from {@link #chars()}. */
    public static CharTokenizer fromChars(char[] chars) {
        if (chars.length < 2 || chars[EOS] != EOS_CHAR) {
            throw new IllegalArgumentException("Not a CharTokenizer vocabulary: id 0 must be the end-of-text character");
        }
        return new CharTokenizer(chars.clone());
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
