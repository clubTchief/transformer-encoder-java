package com.example.story;

/** Hyper-parameters for the decoder-only model. */
public record GptConfig(int vocabSize, int dModel, int numHeads, int numLayers,
                        int dFF, int contextLength, float eps) {

    public int headDim() {
        return dModel / numHeads;
    }

    /** ~0.8M parameters. Realistic on a laptop CPU. */
    public static GptConfig tiny(int vocabSize) {
        return new GptConfig(vocabSize, 128, 4, 4, 352, 128, 1e-5f);
    }

    /** ~5M parameters. Wants a GPU or patience. dFF ~ 8/3 * d (SwiGLU convention), multiple of 64. */
    public static GptConfig small(int vocabSize) {
        return new GptConfig(vocabSize, 256, 8, 6, 704, 256, 1e-5f);
    }
}
