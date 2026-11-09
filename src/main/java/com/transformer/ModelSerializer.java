package com.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.io.IOException;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class ModelSerializer {

    /**
     * Serializes model parameter tensors into a binary checkpoint file.
     */
    public static void saveWeights(String filePath, List<NDArray> trainableParams) throws IOException {
        Path path = Paths.get(filePath);
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }

        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filePath)))) {
            dos.writeInt(trainableParams.size());

            for (NDArray param : trainableParams) {
                Shape shape = param.getShape();
                long[] shapeArray = shape.getShape();

                // Write tensor rank
                dos.writeInt(shapeArray.length);

                // Write shape dimensions
                for (long dim : shapeArray) {
                    dos.writeLong(dim);
                }

                // Write float values
                float[] data = param.toFloatArray();
                dos.writeInt(data.length);
                for (float val : data) {
                    dos.writeFloat(val);
                }
            }
        }
        System.out.println(" Successfully saved " + trainableParams.size() + " parameter tensors to: " + filePath);
    }

    /**
     * Loads saved binary float weights directly into existing model parameter tensors.
     */
    public static void loadWeights(NDManager manager, String filePath, List<NDArray> targetParams) throws IOException {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) {
            int numTensors = dis.readInt();

            if (numTensors != targetParams.size()) {
                throw new IllegalArgumentException("Checkpoint tensor count (" + numTensors + 
                        ") does not match target model parameter count (" + targetParams.size() + ")!");
            }

            for (int i = 0; i < numTensors; i++) {
                int rank = dis.readInt();
                long[] shapeArray = new long[rank];
                for (int r = 0; r < rank; r++) {
                    shapeArray[r] = dis.readLong();
                }

                int dataLen = dis.readInt();
                float[] data = new float[dataLen];
                for (int d = 0; d < dataLen; d++) {
                    data[d] = dis.readFloat();
                }

                long[] expected = targetParams.get(i).getShape().getShape();
                if (!java.util.Arrays.equals(shapeArray, expected)) {
                    throw new IllegalArgumentException("Checkpoint tensor #" + i + " has shape "
                            + java.util.Arrays.toString(shapeArray) + " but the model expects "
                            + java.util.Arrays.toString(expected) + " (different architecture?)");
                }

                // Set raw float data directly into target tensor
                targetParams.get(i).set(data);
            }
        }
        System.out.println(" Successfully loaded " + targetParams.size() + " parameter tensors from: " + filePath);
    }
}