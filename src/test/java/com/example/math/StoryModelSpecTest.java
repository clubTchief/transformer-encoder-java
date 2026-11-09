package com.example.math;

import ai.djl.ndarray.NDManager;
import com.dictionary.StoryDataset;
import com.dictionary.StoryModelSpec;
import com.dictionary.Vocabulary;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class StoryModelSpecTest {

    @Test
    @DisplayName("Every training sentence fits in ONE window, so no sentence is cut into context-free chunks")
    void testEverySentenceIsOneSample() {
        Vocabulary vocab = StoryModelSpec.vocabulary();
        StoryDataset dataset = new StoryDataset(StoryModelSpec.CORPUS, vocab, StoryModelSpec.SEQ_LEN);
        assertEquals(StoryModelSpec.CORPUS.size(), dataset.size(),
                "more samples than sentences means a sentence was split; raise SEQ_LEN");
    }

    @Test
    @DisplayName("Every sample ends in EOS inside a real context, so the model can learn to finish a sentence")
    void testEverySampleLearnsEos() {
        Vocabulary vocab = StoryModelSpec.vocabulary();
        StoryDataset dataset = new StoryDataset(StoryModelSpec.CORPUS, vocab, StoryModelSpec.SEQ_LEN);
        int eos = vocab.getId(Vocabulary.EOS_TOKEN);
        int dot = vocab.getId(".");
        int seqLen = StoryModelSpec.SEQ_LEN;

        try (NDManager manager = NDManager.newBaseManager()) {
            StoryDataset.Batch batch = dataset.getBatch(manager, 0, dataset.size());
            float[] inputs = batch.inputs.toFloatArray();
            float[] targets = batch.targets.toFloatArray();

            for (int row = 0; row < dataset.size(); row++) {
                int eosAt = -1;
                for (int t = 0; t < seqLen; t++) {
                    if ((int) targets[row * seqLen + t] == eos) {
                        eosAt = t;
                    }
                }
                assertTrue(eosAt > 0, "sample " + row + " must predict EOS after real context, not at position 0");
                assertEquals(dot, (int) inputs[row * seqLen + eosAt], "EOS must be predicted right after the final '.'");
            }
        }
    }
}
