package com.dictionary;

import ai.djl.ndarray.NDManager;
import com.example.story.CharTokenizer;
import com.generation.Sampler;
import com.generation.TextGenerator;
import com.transformer.ModelCheckpoint;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.function.Consumer;

/**
 * Interactive story generator for checkpoints written by CorpusTrainer. The checkpoint carries the architecture, the
 * vocabulary and the training window, so nothing about the training run has to be repeated here.
 *
 * mvn -q exec:java -Dexec.mainClass=com.dictionary.CorpusGeneratorCLI -Dexec.args="model=models/tinystories.ckpt temp=0.8"
 */
public final class CorpusGeneratorCLI {

    private static final Set<String> KNOWN = Set.of("model", "tokens", "temp", "topk", "topp", "seed", "prompt");

    private CorpusGeneratorCLI() {
    }

    public static void main(String[] args) throws IOException {
        Map<String, String> a = Args.parse(args);
        Args.rejectUnknown(a, KNOWN);
        Path path = Path.of(a.getOrDefault("model", "models/tinystories.ckpt"));
        int tokens = Integer.parseInt(a.getOrDefault("tokens", "400"));
        float temperature = Float.parseFloat(a.getOrDefault("temp", "0.8"));
        int topK = Integer.parseInt(a.getOrDefault("topk", "40"));
        float topP = Float.parseFloat(a.getOrDefault("topp", "0.95"));
        long seed = a.containsKey("seed") ? Long.parseLong(a.get("seed")) : System.nanoTime();

        if (!Files.exists(path)) {
            System.err.println("No checkpoint at " + path + ". Train one first with CorpusTrainer.");
            return;
        }
        if (!ModelCheckpoint.isCheckpoint(path)) {
            System.err.println(path + " is not a self-describing checkpoint (legacy word-level files belong to StoryGeneratorCLI).");
            return;
        }

        try (NDManager manager = NDManager.newBaseManager()) {
            ModelCheckpoint.Loaded loaded = ModelCheckpoint.load(manager, path);
            CharTokenizer tokenizer = CharTokenizer.fromChars(loaded.vocabulary());
            int window = Integer.parseInt(loaded.metadata().getOrDefault("seqLen", "128"));

            System.out.println("Loaded " + path + ": " + loaded.model().getConfig() + ", window " + window
                    + ", trained " + loaded.metadata().getOrDefault("steps", "?") + " steps"
                    + ", validation loss " + loaded.metadata().getOrDefault("validationLoss", "?"));

            String oneShot = a.get("prompt");
            if (oneShot != null) {
                complete(manager, loaded, tokenizer, window, oneShot, tokens, temperature, topK, topP, seed, System.out::print);
                System.out.println();
                return;
            }

            System.out.println("Type the start of a story and press Enter (an empty line starts a new story), 'exit' to quit.");
            try (Scanner scanner = new Scanner(System.in)) {
                long run = 0;
                while (true) {
                    System.out.print("\nStory opening> ");
                    if (!scanner.hasNextLine()) {
                        break; // end of input
                    }
                    String line = scanner.nextLine();
                    if (line.trim().equalsIgnoreCase("exit") || line.trim().equalsIgnoreCase("quit")) {
                        break;
                    }
                    String unknown = unknownCharacters(tokenizer, line);
                    if (!unknown.isEmpty()) {
                        System.out.println("(characters the model does not know are skipped: " + unknown + ")");
                    }
                    complete(manager, loaded, tokenizer, window, line, tokens, temperature, topK, topP, seed + run++, System.out::print);
                    System.out.println();
                }
            }
        }
    }

    /**
     * Continues {@code prompt}. Generation starts after an end-of-text token, like every story in the training text,
     * stops when the model emits another one or after {@code maxTokens} characters, and slides its context window so
     * it can run longer than the window the model was trained on.
     *
     * @param sink receives the prompt first and then every generated piece as soon as it exists
     * @return the prompt plus the generated text
     */
    public static String complete(NDManager manager, ModelCheckpoint.Loaded loaded, CharTokenizer tokenizer, int window,
                                  String prompt, int maxTokens, float temperature, int topK, float topP, long seed,
                                  Consumer<String> sink) {
        List<Integer> ids = new ArrayList<>();
        ids.add(CharTokenizer.EOS);
        for (int id : tokenizer.encode(prompt)) {
            if (id != CharTokenizer.UNK) {
                ids.add(id);
            }
        }

        StringBuilder text = new StringBuilder(prompt);
        sink.accept(prompt);
        new TextGenerator(loaded.model(), new Sampler(seed)).generate(manager,
                ids.stream().mapToInt(Integer::intValue).toArray(), maxTokens, CharTokenizer.EOS,
                temperature, topK, topP, 1.0f, window, id -> {
                    String piece = tokenizer.decodeToken(id);
                    text.append(piece);
                    sink.accept(piece);
                });
        return text.toString();
    }

    /** The characters of {@code text} that are not in the vocabulary. */
    static String unknownCharacters(CharTokenizer tokenizer, String text) {
        StringBuilder unknown = new StringBuilder();
        for (char c : text.toCharArray()) {
            if (tokenizer.encode(String.valueOf(c))[0] == CharTokenizer.UNK && unknown.indexOf(String.valueOf(c)) < 0) {
                unknown.append(c);
            }
        }
        return unknown.toString();
    }
}
