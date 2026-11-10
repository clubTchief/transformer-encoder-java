package com.dictionary;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/** key=value command-line arguments. A misspelled key is an error instead of being silently ignored. */
final class Args {

    private Args() {
    }

    static Map<String, String> parse(String[] args) {
        Map<String, String> parsed = new LinkedHashMap<>();
        for (String arg : args) {
            int eq = arg.indexOf('=');
            if (eq <= 0) {
                throw new IllegalArgumentException("Arguments look like key=value but got '" + arg + "'");
            }
            parsed.put(arg.substring(0, eq), arg.substring(eq + 1));
        }
        return parsed;
    }

    static void rejectUnknown(Map<String, String> parsed, Set<String> known) {
        for (String key : parsed.keySet()) {
            if (!known.contains(key)) {
                throw new IllegalArgumentException("Unknown argument '" + key + "'. Known arguments: " + new TreeSet<>(known));
            }
        }
    }
}
