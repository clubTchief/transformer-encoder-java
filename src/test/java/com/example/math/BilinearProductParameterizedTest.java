package com.example.math;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class BilinearProductParameterizedTest {

    private final double delta = 1e-9;

    static Stream<Arguments> provideVectorPairsAndExpectedDotProduct() {
        return Stream.of(
            Arguments.of(new double[]{1.0, 2.0}, new double[]{3.0, 4.0}, 11.0),
            Arguments.of(new double[]{0.0, 0.0}, new double[]{3.0, 4.0}, 0.0),
            Arguments.of(new double[]{1.0, -2.0}, new double[]{2.0, 1.0}, 0.0),
            Arguments.of(new double[]{-1.0, -2.0}, new double[]{-3.0, 4.0}, -5.0),
            Arguments.of(new double[]{1.0, 2.0, 3.0}, new double[]{4.0, 5.0, 6.0}, 32.0)
        );
    }

    @ParameterizedTest(name = "Dot product of x and y should equal {2}")
    @MethodSource("provideVectorPairsAndExpectedDotProduct")
    void testDotProductParameterized(double[] x, double[] y, double expected) {
        assertEquals(expected, BilinearProduct.dotProduct(x, y), delta);
    }
}