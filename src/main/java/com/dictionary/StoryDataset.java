package com.dictionary;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.List;

public class StoryDataset {

    public static class Batch {
        public final NDArray inputs;
        public final NDArray targets;

        public Batch(NDArray inputs, NDArray targets) {
            this.inputs = inputs;
            this.targets = targets;
        }
    }

    private final List<int[]> sequencePairs = new ArrayList<>();
    private final Vocabulary vocab;
    private final int seqLen;

    public StoryDataset(List<String> stories, Vocabulary vocab, int seqLen) {
        this.vocab = vocab;
        this.seqLen = seqLen;

        for (String story : stories) {
            String[] tokens = Vocabulary.tokenize(story);
            if (tokens.length <= 1) continue;

            List<Integer> ids = new ArrayList<>();
            ids.add(vocab.getId(Vocabulary.BOS_TOKEN));
            for (String token : tokens) {
                ids.add(vocab.getId(token));
            }
            ids.add(vocab.getId(Vocabulary.EOS_TOKEN));

            for (int i = 0; i < ids.size() - 1; i += seqLen) {
                int end = Math.min(i + seqLen + 1, ids.size());
                if (end - i < 2) break;

                int[] chunk = new int[seqLen + 1];
                for (int k = 0; k < chunk.length; k++) chunk[k] = vocab.getPadId();

                for (int j = 0; j < end - i; j++) {
                    chunk[j] = ids.get(i + j);
                }
                sequencePairs.add(chunk);
            }
        }
    }

    public int getNumBatches(int batchSize) {
        return (int) Math.ceil((double) sequencePairs.size() / batchSize);
    }

    public Batch getBatch(NDManager manager, int batchIndex, int batchSize) {
        int startIdx = batchIndex * batchSize;
        int endIdx = Math.min(startIdx + batchSize, sequencePairs.size());
        int actualBatchSize = endIdx - startIdx;

        float[] inputData = new float[actualBatchSize * seqLen];
        float[] targetData = new float[actualBatchSize * seqLen];

        for (int b = 0; b < actualBatchSize; b++) {
            int[] chunk = sequencePairs.get(startIdx + b);
            for (int t = 0; t < seqLen; t++) {
                inputData[b * seqLen + t] = chunk[t];
                targetData[b * seqLen + t] = chunk[t + 1];
            }
        }

        // Corrected DJL array instantiation: create(float[]) followed by reshape()
        NDArray inputs = manager.create(inputData).reshape(new Shape(actualBatchSize, seqLen));
        NDArray targets = manager.create(targetData).reshape(new Shape(actualBatchSize, seqLen));

        return new Batch(inputs, targets);
    }

    public int size() {
        return sequencePairs.size();
    }
}