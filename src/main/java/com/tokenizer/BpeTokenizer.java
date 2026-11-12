package com.tokenizer;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Byte-level byte-pair encoding (the scheme of GPT-2): the text is cut into words, every word becomes its UTF-8
 * bytes, and the most frequent neighbouring pair of tokens is merged again and again. Frequent words end up as a
 * single token, rare words as a few pieces and any text at all (any language, emoji) can be encoded because the 256
 * bytes are always available.
 *
 * Ids: 0 = end of text, 1..256 = the bytes 0..255, 257 + k = the k-th learned merge.
 * Train one with {@link BpeTrainer}.
 */
public final class BpeTokenizer implements Tokenizer {

    public static final int EOS = 0;
    static final int BYTE_BASE = 1;
    static final int FIRST_MERGE_ID = 257;

    private static final int MAGIC = 0x42504531; // "BPE1"
    private static final int CACHE_LIMIT = 500_000;

    /** GPT-2's word pattern: merges never cross a word boundary, and a leading space belongs to the word after it. */
    static final Pattern WORDS = Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    private final int[] left;   // merge k joins left[k] and right[k] into the token 257 + k
    private final int[] right;
    private final Map<Long, Integer> ranks = new HashMap<>(); // pair -> k
    private final byte[][] tokenBytes;
    private final Map<String, int[]> cache = new ConcurrentHashMap<>();

    BpeTokenizer(int[] left, int[] right) {
        if (left.length != right.length) {
            throw new IllegalArgumentException("left and right merge lists differ in length");
        }
        this.left = left.clone();
        this.right = right.clone();
        int merges = left.length;
        this.tokenBytes = new byte[FIRST_MERGE_ID + merges][];
        tokenBytes[EOS] = END_OF_TEXT.getBytes(StandardCharsets.UTF_8);
        for (int b = 0; b < 256; b++) {
            tokenBytes[BYTE_BASE + b] = new byte[]{(byte) b};
        }
        for (int k = 0; k < merges; k++) {
            int id = FIRST_MERGE_ID + k;
            if (left[k] <= EOS || right[k] <= EOS || left[k] >= id || right[k] >= id) {
                throw new IllegalArgumentException("Merge " + k + " refers to a token that does not exist yet");
            }
            byte[] a = tokenBytes[left[k]];
            byte[] b = tokenBytes[right[k]];
            byte[] joined = Arrays.copyOf(a, a.length + b.length);
            System.arraycopy(b, 0, joined, a.length, b.length);
            tokenBytes[id] = joined;
            if (ranks.putIfAbsent(pairKey(left[k], right[k]), k) != null) {
                throw new IllegalArgumentException("Merge " + k + " repeats an earlier merge");
            }
        }
    }

    static long pairKey(int a, int b) {
        return ((long) a << 32) | (b & 0xFFFFFFFFL);
    }

    @Override
    public String kind() {
        return "bpe";
    }

    @Override
    public int vocabSize() {
        return tokenBytes.length;
    }

    public int mergeCount() {
        return left.length;
    }

    @Override
    public int eosId() {
        return EOS;
    }

    @Override
    public int unkId() {
        return -1;
    }

    @Override
    public int[] encode(String text) {
        IntList out = new IntList(text.length() / 3 + 8);
        int from = 0;
        while (true) {
            int marker = text.indexOf(END_OF_TEXT, from);
            encodeSegment(marker < 0 ? text.substring(from) : text.substring(from, marker), out);
            if (marker < 0) {
                break;
            }
            out.add(EOS);
            from = marker + END_OF_TEXT.length();
        }
        return out.toArray();
    }

    private void encodeSegment(String segment, IntList out) {
        Matcher words = WORDS.matcher(segment);
        while (words.find()) {
            out.addAll(encodeWord(words.group()));
        }
    }

    /** Applies the learned merges to one word, always the one that was learned first, until none applies. */
    private int[] encodeWord(String word) {
        int[] known = cache.get(word);
        if (known != null) {
            return known;
        }
        byte[] bytes = word.getBytes(StandardCharsets.UTF_8);
        int[] ids = new int[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            ids[i] = BYTE_BASE + (bytes[i] & 0xFF);
        }
        int length = ids.length;
        while (length > 1) {
            int bestRank = Integer.MAX_VALUE;
            for (int i = 0; i + 1 < length; i++) {
                Integer rank = ranks.get(pairKey(ids[i], ids[i + 1]));
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                }
            }
            if (bestRank == Integer.MAX_VALUE) {
                break;
            }
            int a = left[bestRank];
            int b = right[bestRank];
            int merged = FIRST_MERGE_ID + bestRank;
            int written = 0;
            for (int i = 0; i < length; ) {
                if (i + 1 < length && ids[i] == a && ids[i + 1] == b) {
                    ids[written++] = merged;
                    i += 2;
                } else {
                    ids[written++] = ids[i++];
                }
            }
            length = written;
        }
        int[] result = Arrays.copyOf(ids, length);
        if (cache.size() < CACHE_LIMIT) {
            cache.put(word, result);
        }
        return result;
    }

    @Override
    public String decode(int[] ids) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        for (int id : ids) {
            bytes.writeBytes(tokenBytes(id));
        }
        return new String(bytes.toByteArray(), StandardCharsets.UTF_8);
    }

    private byte[] tokenBytes(int id) {
        if (id < 0 || id >= tokenBytes.length) {
            throw new IllegalArgumentException("Token id " + id + " is outside the vocabulary of " + tokenBytes.length);
        }
        return tokenBytes[id];
    }

    @Override
    public StreamDecoder newStreamDecoder() {
        return new StreamDecoder() {
            private final ByteArrayOutputStream pending = new ByteArrayOutputStream();

            @Override
            public String accept(int id) {
                if (id == EOS) {
                    return "";
                }
                pending.writeBytes(tokenBytes(id));
                byte[] all = pending.toByteArray();
                int complete = completePrefix(all);
                pending.reset();
                pending.write(all, complete, all.length - complete);
                return new String(all, 0, complete, StandardCharsets.UTF_8);
            }

            @Override
            public String finish() {
                String rest = new String(pending.toByteArray(), StandardCharsets.UTF_8);
                pending.reset();
                return rest;
            }
        };
    }

    /** Length of the longest prefix of {@code bytes} that does not end in the middle of a UTF-8 character. */
    static int completePrefix(byte[] bytes) {
        int n = bytes.length;
        int i = n - 1;
        int continuation = 0;
        while (i >= 0 && continuation < 3 && (bytes[i] & 0xC0) == 0x80) {
            i--;
            continuation++;
        }
        if (i < 0) {
            return n; // only stray continuation bytes: nothing to wait for
        }
        int lead = bytes[i] & 0xFF;
        int needed = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
        return (n - i) < needed ? i : n;
    }

    @Override
    public byte[] serialize() {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        try (DataOutputStream out = new DataOutputStream(buffer)) {
            out.writeInt(MAGIC);
            out.writeInt(left.length);
            for (int k = 0; k < left.length; k++) {
                out.writeInt(left[k]);
                out.writeInt(right[k]);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return buffer.toByteArray();
    }

    public static BpeTokenizer deserialize(byte[] data) {
        try (DataInputStream in = new DataInputStream(new ByteArrayInputStream(data))) {
            if (in.readInt() != MAGIC) {
                throw new IllegalArgumentException("These bytes are not a BPE tokenizer");
            }
            int merges = in.readInt();
            if (merges < 0 || merges > 10_000_000) {
                throw new IllegalArgumentException("Damaged BPE tokenizer: " + merges + " merges");
            }
            int[] left = new int[merges];
            int[] right = new int[merges];
            for (int k = 0; k < merges; k++) {
                left[k] = in.readInt();
                right[k] = in.readInt();
            }
            return new BpeTokenizer(left, right);
        } catch (IOException e) {
            throw new IllegalArgumentException("Damaged BPE tokenizer data: " + e.getMessage(), e);
        }
    }

    /** A growable int array without boxing. */
    private static final class IntList {
        private int[] values;
        private int size;

        IntList(int capacity) {
            values = new int[Math.max(capacity, 8)];
        }

        void add(int value) {
            if (size == values.length) {
                values = Arrays.copyOf(values, values.length * 2);
            }
            values[size++] = value;
        }

        void addAll(int[] more) {
            for (int value : more) {
                add(value);
            }
        }

        int[] toArray() {
            return Arrays.copyOf(values, size);
        }
    }
}
