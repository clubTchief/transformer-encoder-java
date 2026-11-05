package com.example.math;

public class BilinearProduct {

    /**
     * Computes matrix bilinear form Ω(x, y) = x^T * A * y for n-dimensional vectors.
     */
    public static double compute(double[] x, double[] y, double[][] A) {
        validateInputs(x, y, A);
        int n = x.length;
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sum += x[i] * A[i][j] * y[j];
            }
        }
        return sum;
    }

    /**
     * Standard Euclidean Dot Product for n-dimensional vectors.
     */
    public static double dotProduct(double[] x, double[] y) {
        validateVectors(x, y);
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            sum += x[i] * y[i];
        }
        return sum;
    }

    public static double weightedProduct(double[] x, double[] y) {
        double[][] A = {{3.0, 0.0}, {0.0, 2.0}};
        return compute(x, y, A);
    }

    public static double nonPositiveDefiniteForm(double[] x, double[] y) {
        double[][] B = {{1.0, 0.0}, {0.0, -1.0}};
        return compute(x, y, B);
    }

    public static double asymmetricForm(double[] x, double[] y) {
        double[][] M = {{1.0, 2.0}, {0.0, 0.0}};
        return compute(x, y, M);
    }

    private static void validateVectors(double[] x, double[] y) {
        if (x == null || y == null) {
            throw new IllegalArgumentException("Vectors cannot be null.");
        }
        if (x.length != y.length) {
            throw new IllegalArgumentException("Vector dimensions must match.");
        }
    }

    private static void validateInputs(double[] x, double[] y, double[][] A) {
        validateVectors(x, y);
        if (A == null || A.length != x.length) {
            throw new IllegalArgumentException("Square matrix must match vector dimension.");
        }
    }
}