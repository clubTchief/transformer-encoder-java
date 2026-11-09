package com.dictionary;

import com.transformer.LMConfig;

import java.util.List;

/**
 * Everything the trainer, the pipeline and the CLI must agree on. It used to be copy-pasted into each of them,
 * which is how the constructor arguments ended up in different orders; now there is one definition.
 */
public final class StoryModelSpec {

    public static final List<String> CORPUS = List.of(
            "Once upon a time in a distant realm lived a brave knight .",
            "The knight found a mysterious glowing sword inside a dark cave .",
            "A powerful dragon protected the ancient castle in the realm .",
            "The brave knight defended the realm and restored peace to all ."
    );

    /**
     * Training window. It must be at least as long as the longest sentence (BOS + words + EOS - 1 = 14 here), otherwise
     * StoryDataset cuts the sentence into separate chunks and the tail ("." and EOS) is learned without any context,
     * so the model never learns to END a sentence after its words. StoryModelSpecTest guards this.
     */
    public static final int SEQ_LEN = 14;
    public static final int D_MODEL = 64;
    public static final int NUM_HEADS = 4;
    public static final int NUM_LAYERS = 2;
    public static final int D_FF = 256;
    public static final float EPS = 1e-5f;

    public static final String CHECKPOINT_PATH = "models/story_model.bin";

    private StoryModelSpec() {
    }

    public static Vocabulary vocabulary() {
        return new Vocabulary(CORPUS);
    }

    public static LMConfig config(int vocabSize) {
        return new LMConfig(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, vocabSize, EPS);
    }
}
