package com.example.math;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import com.transformer.GradientEngine;
import com.transformer.RopeEmbeddings;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/** RopeEmbeddings is the one RoPE implementation: rotate-half pairing, tables with two identical halves. */
class RopeEmbeddingsTest {

    private NDManager manager;

    @BeforeEach
    void setUp() {
        manager = NDManager.newBaseManager();
        Engine.getInstance().setRandomSeed(1234);
    }

    @AfterEach
    void tearDown() {
        manager.close();
    }

    @Test
    @DisplayName("Tables are [maxLen, headDim]; position 0 is the identity rotation")
    void testTablesShapeAndPositionZero() {
        NDArray[] t = RopeEmbeddings.computeCosSin(manager, 6, 8);
        assertEquals(new Shape(6, 8), t[0].getShape());
        assertEquals(new Shape(6, 8), t[1].getShape());
        float[] cos = t[0].toFloatArray();
        float[] sin = t[1].toFloatArray();
        for (int i = 0; i < 8; i++) {
            assertEquals(1.0f, cos[i], 1e-6f);
            assertEquals(0.0f, sin[i], 1e-6f);
        }
    }

    @Test
    @DisplayName("Convention guard: every table row is two identical halves (what rotate-half pairing expects)")
    void testTablesRepeatFrequenciesAsTwoHalves() {
        int dK = 8, half = dK / 2;
        NDArray[] t = RopeEmbeddings.computeCosSin(manager, 5, dK);
        for (NDArray table : t) {
            float[] v = table.toFloatArray();
            for (int pos = 0; pos < 5; pos++) {
                for (int i = 0; i < half; i++) {
                    assertEquals(v[pos * dK + i], v[pos * dK + i + half], 1e-6f, "pos " + pos + " channel " + i);
                }
            }
        }
    }

    @Test
    @DisplayName("An odd head dimension is rejected")
    void testOddHeadDimRejected() {
        assertThrows(IllegalArgumentException.class, () -> RopeEmbeddings.computeCosSin(manager, 4, 7));
    }

    @Test
    @DisplayName("slice() returns exactly the table rows of the requested absolute positions")
    void testSliceReturnsRequestedRows() {
        int dK = 8;
        NDArray[] t = RopeEmbeddings.computeCosSin(manager, 7, dK);
        NDArray[] s = RopeEmbeddings.slice(t[0], t[1], 2, 3);
        assertEquals(new Shape(3, dK), s[0].getShape());
        assertArrayEquals(Arrays.copyOfRange(t[0].toFloatArray(), 2 * dK, 5 * dK), s[0].toFloatArray(), 1e-6f);
        assertArrayEquals(Arrays.copyOfRange(t[1].toFloatArray(), 2 * dK, 5 * dK), s[1].toFloatArray(), 1e-6f);
    }

    @Test
    @DisplayName("slice() rejects positions beyond the table")
    void testSliceOutOfRange() {
        NDArray[] t = RopeEmbeddings.computeCosSin(manager, 4, 8);
        assertThrows(IllegalArgumentException.class, () -> RopeEmbeddings.slice(t[0], t[1], 3, 2));
        assertThrows(IllegalArgumentException.class, () -> RopeEmbeddings.slice(t[0], t[1], -1, 1));
        assertThrows(IllegalArgumentException.class, () -> RopeEmbeddings.slice(t[0], t[1], 0, 0));
    }

    @Test
    @DisplayName("applyRope only accepts [batch, heads, seq, headDim]")
    void testApplyRequiresFourDimensions() {
        NDArray[] t = RopeEmbeddings.computeCosSin(manager, 4, 8);
        NDArray threeD = manager.ones(new Shape(2, 4, 8));
        assertThrows(IllegalArgumentException.class, () -> RopeEmbeddings.applyRope(threeD, t[0], t[1]));
    }

    @Test
    @DisplayName("applyRope matches a straightforward loop implementation of rotate-half")
    void testApplyMatchesReferenceLoop() {
        int b = 2, h = 3, n = 5, dK = 8, half = dK / 2;
        NDArray x = manager.randomNormal(0.0f, 1.0f, new Shape(b, h, n, dK), DataType.FLOAT32);
        NDArray[] t = RopeEmbeddings.computeCosSin(manager, n, dK);

        float[] in = x.toFloatArray();
        float[] ref = new float[in.length];
        for (int bi = 0; bi < b; bi++) {
            for (int hi = 0; hi < h; hi++) {
                for (int pos = 0; pos < n; pos++) {
                    int base = ((bi * h + hi) * n + pos) * dK;
                    for (int i = 0; i < half; i++) {
                        double th = pos / Math.pow(10000.0, (2.0 * i) / dK);
                        float c = (float) Math.cos(th), s = (float) Math.sin(th);
                        float lo = in[base + i], hi2 = in[base + i + half];
                        ref[base + i] = lo * c - hi2 * s;
                        ref[base + i + half] = hi2 * c + lo * s;
                    }
                }
            }
        }
        assertArrayEquals(ref, RopeEmbeddings.applyRope(x, t[0], t[1]).toFloatArray(), 1e-5f);
    }

    @Test
    @DisplayName("applyRope is differentiable: the gradient of sum(rotation) equals the analytic derivative")
    void testGradientMatchesAnalyticValues() {
        int h = 2, n = 4, dK = 8, half = dK / 2;
        NDArray x = manager.ones(new Shape(1, h, n, dK), DataType.FLOAT32);
        x.setRequiresGradient(true);
        NDArray[] t = RopeEmbeddings.computeCosSin(manager, n, dK);

        try (GradientCollector collector = new GradientEngine(0.01f).createCollector()) {
            collector.backward(RopeEmbeddings.applyRope(x, t[0], t[1]).sum());
        }

        // out[i] = x[i]c - x[i+half]s (first half), out[i+half] = x[i+half]c + x[i]s  =>  d/dx[i] = c + s, d/dx[i+half] = c - s
        float[] g = x.getGradient().toFloatArray();
        int idx = 0;
        for (int head = 0; head < h; head++) {
            for (int pos = 0; pos < n; pos++) {
                double[] expected = new double[dK];
                for (int j = 0; j < half; j++) {
                    double th = pos / Math.pow(10000.0, (2.0 * j) / dK);
                    expected[j] = Math.cos(th) + Math.sin(th);
                    expected[j + half] = Math.cos(th) - Math.sin(th);
                }
                for (int c = 0; c < dK; c++) {
                    assertEquals(expected[c], g[idx++], 1e-5, "head " + head + " pos " + pos + " channel " + c);
                }
            }
        }
    }

    /** score of q at position m against k at position n, each rotated by its own absolute position */
    private float score(NDArray q, NDArray k, NDArray[] tables, int m, int n) {
        NDArray[] rm = RopeEmbeddings.slice(tables[0], tables[1], m, 1);
        NDArray[] rn = RopeEmbeddings.slice(tables[0], tables[1], n, 1);
        NDArray qr = RopeEmbeddings.applyRope(q, rm[0], rm[1]);
        NDArray kr = RopeEmbeddings.applyRope(k, rn[0], rn[1]);
        return qr.mul(kr).sum().getFloat();
    }

    @Test
    @DisplayName("Defining property: the attention score depends only on the RELATIVE distance m - n")
    void testScoreDependsOnlyOnRelativePosition() {
        int dK = 8;
        NDArray[] tables = RopeEmbeddings.computeCosSin(manager, 32, dK);
        NDArray q = manager.randomNormal(0.0f, 1.0f, new Shape(1, 1, 1, dK), DataType.FLOAT32);
        NDArray k = manager.randomNormal(0.0f, 1.0f, new Shape(1, 1, 1, dK), DataType.FLOAT32);

        float distance2a = score(q, k, tables, 5, 3);
        float distance2b = score(q, k, tables, 21, 19);
        float distance1 = score(q, k, tables, 5, 4);
        float distance0 = score(q, k, tables, 7, 7);

        assertEquals(distance2a, distance2b, 1e-4f, "same distance, different absolute positions");
        assertTrue(Math.abs(distance2a - distance1) > 1e-3f, "different distances must give different scores");
        assertEquals(q.mul(k).sum().getFloat(), distance0, 1e-4f, "distance 0 is a plain dot product");
    }
}
