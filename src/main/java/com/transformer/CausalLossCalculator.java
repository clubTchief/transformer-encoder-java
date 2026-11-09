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

        // 4. Log-probability of each target token: shape [batchSize * seqLen].
        //    (torch.gather needs an index with the same rank as its input, so a 1-D index on this 2-D tensor fails with
        //    "Index tensor must have the same number of dimensions as input tensor". Selecting with a one-hot mask is
        //    the same maths, differentiable, and the approach the other trainers in this repo already use.)
        NDArray targetMask = flatTargets.toType(DataType.INT64, false).oneHot((int) vocabSize); // [N, vocabSize]
        NDArray targetLogProbs = logProbs.mul(targetMask).sum(new int[]{1});

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