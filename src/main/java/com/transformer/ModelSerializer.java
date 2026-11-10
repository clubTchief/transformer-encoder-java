package com.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class ModelSerializer {

    /**
     * Serializes model parameter tensors into a binary checkpoint file (tensors only: no architecture, no vocabulary;
     * see {@link ModelCheckpoint} for the self-describing format).
     */
    public static void saveWeights(String filePath, List<NDArray> trainableParams) throws IOException {
        Path path = Paths.get(filePath);
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }

        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filePath)))) {
            writeTensors(dos, trainableParams);
        }
        System.out.println(" Successfully saved " + trainableParams.size() + " parameter tensors to: " + filePath);
    }

    /**
     * Loads saved binary float weights directly into existing model parameter tensors.
     */
    public static void loadWeights(NDManager manager, String filePath, List<NDArray> targetParams) throws IOException {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) {
            readTensors(dis, targetParams);
        }
        System.out.println(" Successfully loaded " + targetParams.size() + " parameter tensors from: " + filePath);
    }

    /** Writes: tensor count, then per tensor its rank, its dimensions and its float values. */
    public static void writeTensors(DataOutputStream dos, List<NDArray> params) throws IOException {
        dos.writeInt(params.size());
        for (NDArray param : params) {
            long[] shapeArray = param.getShape().getShape();
            dos.writeInt(shapeArray.length);
            for (long dim : shapeArray) {
                dos.writeLong(dim);
            }
            float[] data = param.toFloatArray();
            dos.writeInt(data.length);
            for (float val : data) {
                dos.writeFloat(val);
            }
        }
    }

    /**
     * Reads what {@link #writeTensors} wrote into existing tensors. Throws IllegalArgumentException when the count or a
     * shape does not match the target (a checkpoint of another architecture) and IOException for damaged data.
     */
    public static void readTensors(DataInputStream dis, List<NDArray> targetParams) throws IOException {
        int numTensors = dis.readInt();
        if (numTensors != targetParams.size()) {
            throw new IllegalArgumentException("Checkpoint tensor count (" + numTensors
                    + ") does not match target model parameter count (" + targetParams.size() + ")!");
        }

        for (int i = 0; i < numTensors; i++) {
            int rank = dis.readInt();
            if (rank < 0 || rank > 8) {
                throw new IOException("Damaged checkpoint: tensor #" + i + " claims rank " + rank);
            }
            long[] shapeArray = new long[rank];
            for (int r = 0; r < rank; r++) {
                shapeArray[r] = dis.readLong();
            }

            long[] expected = targetParams.get(i).getShape().getShape();
            if (!Arrays.equals(shapeArray, expected)) {
                throw new IllegalArgumentException("Checkpoint tensor #" + i + " has shape "
                        + Arrays.toString(shapeArray) + " but the model expects "
                        + Arrays.toString(expected) + " (different architecture?)");
            }

            int dataLen = dis.readInt();
            if (dataLen != new Shape(shapeArray).size()) {
                throw new IOException("Damaged checkpoint: tensor #" + i + " has " + dataLen + " values for shape "
                        + Arrays.toString(shapeArray));
            }
            float[] data = new float[dataLen];
            for (int d = 0; d < dataLen; d++) {
                data[d] = dis.readFloat();
            }
            targetParams.get(i).set(data);
        }
    }
}
