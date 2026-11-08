package com.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class CausalLossCalculator {

    public static NDArray calculateLoss(NDManager manager, NDArray logits, NDArray targets, int padTokenId) {
        Shape logitsShape = logits.getShape();
        long batchSize = logitsShape.get(0);
        long seqLen = logitsShape.get(1);
        long vocabSize = logitsShape.get(2);

        // 1. Flatten logits to 2D [batchSize * seqLen, vocabSize]
        NDArray flatLogits = logits.reshape(new Shape(batchSize * seqLen, vocabSize));

        // 2. Compute log-softmax along vocabulary dimension
        NDArray logProbs = flatLogits.logSoftmax(1);

        // 3. Flatten targets to 1D [batchSize * seqLen]
        NDArray flatTargets = targets.reshape(new Shape(batchSize * seqLen));

        // 4. Gather log-probabilities for target indices: shape [batchSize * seqLen]
        NDArray targetLogProbs = logProbs.gather(flatTargets.toType(DataType.INT64, false), 1);

        // 5. Create mask to exclude <PAD> tokens (using neq)
        NDArray nonPadMask = flatTargets.neq(padTokenId).toType(DataType.FLOAT32, false);

        // 6. Compute negative log-likelihood (NLL) masked by non-padding tokens
        NDArray nll = targetLogProbs.neg().mul(nonPadMask);

        // 7. Return average loss over valid (non-padded) tokens
        NDArray totalValidTokens = nonPadMask.sum();
        NDArray safeTokenCount = totalValidTokens.maximum(manager.create(1.0f));

        return nll.sum().div(safeTokenCount);
    }

    public static NDArray calculateLoss(NDManager manager, NDArray logits, NDArray targets) {
        return calculateLoss(manager, logits, targets, 0);
    }
}