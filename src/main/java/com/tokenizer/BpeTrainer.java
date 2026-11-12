package com.tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;

/**
 * Learns the merges of a {@link BpeTokenizer} from a text. The text is reduced to its distinct words with their counts;
 * then the most frequent pair of neighbouring tokens (over all words, weighted by count) is merged into a new token,
 * and the pair statistics of the words that contained it are updated, until the vocabulary has the wanted size or no
 * pair occurs twice any more. Ties are broken by the smaller pair, so the same text always gives the same tokenizer.
 */
public final class BpeTrainer {

    /** Reports the progress of {@link #train}. */
    public interface Progress {
        /** Called every 100 merges and once more when training ends (merges may then be below totalMerges). */
        void update(int merges, int totalMerges, long countOfLastPair);
    }

    private BpeTrainer() {
    }

    /**
     * @param vocabSize wanted vocabulary size, more than 257 (end of text + 256 bytes); fewer merges are learned if the
     *                  text runs out of repeating pairs
     * @param progress  may be null
     */
    public static BpeTokenizer train(String text, int vocabSize, Progress progress) {
        if (vocabSize <= BpeTokenizer.FIRST_MERGE_ID) {
            throw new IllegalArgumentException("vocabSize must be larger than " + BpeTokenizer.FIRST_MERGE_ID
                    + " (end of text + 256 bytes) but was " + vocabSize);
        }
        int targetMerges = vocabSize - BpeTokenizer.FIRST_MERGE_ID;

        // 1. The distinct words and how often each one occurs
        Map<String, Long> wordCounts = new HashMap<>();
        int from = 0;
        while (true) {
            int marker = text.indexOf(Tokenizer.END_OF_TEXT, from);
            String segment = marker < 0 ? text.substring(from) : text.substring(from, marker);
            Matcher words = BpeTokenizer.WORDS.matcher(segment);
            while (words.find()) {
                wordCounts.merge(words.group(), 1L, Long::sum);
            }
            if (marker < 0) {
                break;
            }
            from = marker + Tokenizer.END_OF_TEXT.length();
        }

        // 2. Every word as a sequence of byte tokens
        int n = wordCounts.size();
        int[][] words = new int[n][];
        long[] frequency = new long[n];
        int index = 0;
        for (Map.Entry<String, Long> entry : wordCounts.entrySet()) {
            byte[] bytes = entry.getKey().getBytes(StandardCharsets.UTF_8);
            int[] ids = new int[bytes.length];
            for (int i = 0; i < bytes.length; i++) {
                ids[i] = BpeTokenizer.BYTE_BASE + (bytes[i] & 0xFF);
            }
            words[index] = ids;
            frequency[index] = entry.getValue();
            index++;
        }

        // 3. Pair statistics: how often each pair occurs, and which words contain it
        Map<Long, Long> pairCounts = new HashMap<>();
        Map<Long, Set<Integer>> pairWords = new HashMap<>();
        for (int w = 0; w < n; w++) {
            addWord(words[w], w, frequency[w], pairCounts, pairWords);
        }

        // 4. Merge the most frequent pair until the vocabulary is big enough
        int[] left = new int[targetMerges];
        int[] right = new int[targetMerges];
        int done = 0;
        long lastCount = 0;
        while (done < targetMerges) {
            long bestKey = -1;
            long bestCount = 0;
            for (Map.Entry<Long, Long> entry : pairCounts.entrySet()) {
                long count = entry.getValue();
                long key = entry.getKey();
                if (count > bestCount || (count == bestCount && key < bestKey)) {
                    bestCount = count;
                    bestKey = key;
                }
            }
            if (bestCount < 2) {
                break; // nothing repeats any more: further merges would only memorise single occurrences
            }

            int a = (int) (bestKey >>> 32);
            int b = (int) bestKey;
            int newToken = BpeTokenizer.FIRST_MERGE_ID + done;
            left[done] = a;
            right[done] = b;

            for (int w : new ArrayList<>(pairWords.get(bestKey))) {
                removeWord(words[w], w, frequency[w], pairCounts, pairWords);
                words[w] = merge(words[w], a, b, newToken);
                addWord(words[w], w, frequency[w], pairCounts, pairWords);
            }

            done++;
            lastCount = bestCount;
            if (progress != null && done % 100 == 0) {
                progress.update(done, targetMerges, bestCount);
            }
        }
        if (progress != null && done % 100 != 0) {
            progress.update(done, targetMerges, lastCount); // the final state, also when the text ran out of repeating pairs early
        }
        return new BpeTokenizer(Arrays.copyOf(left, done), Arrays.copyOf(right, done));
    }

    private static void addWord(int[] word, int index, long frequency, Map<Long, Long> counts, Map<Long, Set<Integer>> where) {
        for (int i = 0; i + 1 < word.length; i++) {
            long key = BpeTokenizer.pairKey(word[i], word[i + 1]);
            counts.merge(key, frequency, Long::sum);
            where.computeIfAbsent(key, k -> new HashSet<>()).add(index);
        }
    }

    private static void removeWord(int[] word, int index, long frequency, Map<Long, Long> counts, Map<Long, Set<Integer>> where) {
        for (int i = 0; i + 1 < word.length; i++) {
            long key = BpeTokenizer.pairKey(word[i], word[i + 1]);
            Long count = counts.get(key);
            if (count != null) {
                long remaining = count - frequency;
                if (remaining <= 0) {
                    counts.remove(key);
                } else {
                    counts.put(key, remaining);
                }
            }
            Set<Integer> containing = where.get(key);
            if (containing != null) {
                containing.remove(index);
                if (containing.isEmpty()) {
                    where.remove(key);
                }
            }
        }
    }

    /** Replaces every (a, b), scanning from the left, by the new token. */
    private static int[] merge(int[] word, int a, int b, int newToken) {
        int[] out = new int[word.length];
        int length = 0;
        for (int i = 0; i < word.length; ) {
            if (i + 1 < word.length && word[i] == a && word[i + 1] == b) {
                out[length++] = newToken;
                i += 2;
            } else {
                out[length++] = word[i++];
            }
        }
        return length == word.length ? out : Arrays.copyOf(out, length);
    }
}
