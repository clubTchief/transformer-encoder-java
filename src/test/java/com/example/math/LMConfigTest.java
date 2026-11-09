package com.example.math;

import com.transformer.LMConfig;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LMConfigTest {

    @Test
    @DisplayName("A valid config reports its head dimension")
    void testHeadDim() {
        assertEquals(16, new LMConfig(2, 64, 4, 256, 100, 1e-5f).headDim());
    }

    @Test
    @DisplayName("Sizes must be positive")
    void testPositiveSizes() {
        assertThrows(IllegalArgumentException.class, () -> new LMConfig(0, 64, 4, 256, 100, 1e-5f));
        assertThrows(IllegalArgumentException.class, () -> new LMConfig(2, 64, 4, 256, 0, 1e-5f));
    }

    @Test
    @DisplayName("dModel must be divisible by numHeads, and the head dimension must be even (RoPE)")
    void testDivisibilityAndEvenHeadDim() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> new LMConfig(2, 30, 4, 64, 10, 1e-5f));
        assertTrue(e.getMessage().contains("divisible"));
        assertThrows(IllegalArgumentException.class, () -> new LMConfig(2, 12, 4, 64, 10, 1e-5f)); // head dim 3
    }
}
