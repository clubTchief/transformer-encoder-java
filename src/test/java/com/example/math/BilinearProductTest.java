package com.example.math;

import com.transformer.BilinearProduct;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class BilinearProductTest {

    private final double delta = 1e-9;

    @Test
    @DisplayName("Dot Product - Symmetry & Positive Definiteness")
    void testDotProduct() {
        double[] x = {1.0, 2.0, 3.0};
        double[] y = {4.0, 5.0, 6.0};
        double[] zero = {0.0, 0.0, 0.0};

        assertEquals(32.0, BilinearProduct.dotProduct(x, y), delta);
        assertEquals(BilinearProduct.dotProduct(x, y), BilinearProduct.dotProduct(y, x), delta);
        assertTrue(BilinearProduct.dotProduct(x, x) > 0);
        assertEquals(0.0, BilinearProduct.dotProduct(zero, zero), delta);
    }

    @Test
    @DisplayName("Arbitrary Matrix Bilinear Form x^T A y")
    void testArbitraryMatrixBilinearForm() {
        double[] x = {1.0, 2.0};
        double[] y = {3.0, 4.0};
        double[][] A = {{2.0, 1.0}, {0.0, 3.0}};

        assertEquals(34.0, BilinearProduct.compute(x, y, A), delta);
    }

    @Test
    @DisplayName("Non-Positive Definite Form - Fails Positive Definiteness")
    void testNonPositiveDefiniteForm() {
        double[] unitY = {0.0, 1.0};
        double result = BilinearProduct.nonPositiveDefiniteForm(unitY, unitY);
        assertFalse(result > 0, "Form must fail positive definiteness test for [0, 1]");
        assertEquals(-1.0, result, delta);
    }

    @Test
    @DisplayName("Asymmetric Form - Fails Symmetry Property")
    void testAsymmetricForm() {
        double[] x = {1.0, 2.0};
        double[] y = {3.0, 4.0};
        assertNotEquals(BilinearProduct.asymmetricForm(x, y), BilinearProduct.asymmetricForm(y, x), delta);
    }
}