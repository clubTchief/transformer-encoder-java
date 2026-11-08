package com.example.math;

import com.transformer.BilinearProduct;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.println("--- Vector Inner Product Calculator ---");
        
        double[] x = readVector(scanner, "x");
        double[] y = readVector(scanner, "y");

        System.out.println("\n--- Calculations ---");
        System.out.printf("1. Standard Dot Product <x, y>: %.4f%n", BilinearProduct.dotProduct(x, y));
        System.out.printf("   Symmetry Check <y, x>:       %.4f%n", BilinearProduct.dotProduct(y, x));
        System.out.printf("   Self Product <x, x>:          %.4f%n", BilinearProduct.dotProduct(x, x));

        System.out.printf("%n2. Weighted Product Ω(x, y):    %.4f%n", BilinearProduct.weightedProduct(x, y));
        System.out.printf("   Self Product Ω(x, x):         %.4f%n", BilinearProduct.weightedProduct(x, x));

        System.out.printf("%n3. Non-PD Form Ω(x, y):          %.4f%n", BilinearProduct.nonPositiveDefiniteForm(x, y));
        System.out.printf("   Self Product Ω(x, x):         %.4f%n", BilinearProduct.nonPositiveDefiniteForm(x, x));

        System.out.printf("%n4. Asymmetric Form Ω(x, y):      %.4f%n", BilinearProduct.asymmetricForm(x, y));
        System.out.printf("   Reversed Form Ω(y, x):        %.4f%n", BilinearProduct.asymmetricForm(y, x));

        scanner.close();
    }

    private static double[] readVector(Scanner scanner, String name) {
        System.out.printf("Enter 2D vector %s (space-separated, e.g., '1 2'): ", name);
        double v1 = scanner.nextDouble();
        double v2 = scanner.nextDouble();
        return new double[]{v1, v2};
    }
}