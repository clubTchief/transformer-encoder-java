package com.example.story;

import java.util.HashMap;
import java.util.Map;

/** Tiny key=value argument parser: corpus=data/stories.txt steps=5000 */
final class Cli {
    private Cli() {}

    static Map<String, String> parse(String[] args) {
        Map<String, String> m = new HashMap<>();
        for (String a : args) {
            int i = a.indexOf('=');
            if (i > 0) m.put(a.substring(0, i), a.substring(i + 1));
        }
        return m;
    }

    static String require(Map<String, String> m, String key) {
        String v = m.get(key);
        if (v == null) throw new IllegalArgumentException("Missing argument: " + key + "=...");
        return v;
    }
}
