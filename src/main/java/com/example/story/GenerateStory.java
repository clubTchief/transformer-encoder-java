package com.example.story;

import ai.djl.ndarray.NDManager;

import java.nio.file.Path;
import java.util.Map;
import java.util.Scanner;

/**
 * mvn -q exec:java -Dexec.mainClass=com.example.story.GenerateStory -Dexec.args="model=checkpoints"
 * Type a story opening (or press Enter for a free-running story). Ctrl-D to quit.
 */
public final class GenerateStory {
    public static void main(String[] args) throws Exception {
        Map<String, String> a = Cli.parse(args);
        Path dir = Path.of(a.getOrDefault("model", "checkpoints"));
        int tokens = Integer.parseInt(a.getOrDefault("tokens", "600"));
        float temp = Float.parseFloat(a.getOrDefault("temp", "0.8"));
        int topK = Integer.parseInt(a.getOrDefault("topk", "40"));
        float topP = Float.parseFloat(a.getOrDefault("topp", "0.95"));

        CharTokenizer tok = CharTokenizer.load(dir.resolve("vocab.txt"));
        try (NDManager manager = NDManager.newBaseManager();
             Scanner in = new Scanner(System.in)) {
            // arch=gpt (default): TinyGpt checkpoint; arch=stack: BlockStackLM (DecoderBlock-based) checkpoint
            StoryGenerator gen = "stack".equals(a.getOrDefault("arch", "gpt"))
                    ? BlockStackLM.load(manager, dir).generator(manager, tok)
                    : new StoryGenerator(manager, TinyGpt.load(manager, dir), tok);
            System.out.print("Story opening> ");
            while (in.hasNextLine()) {
                String prompt = in.nextLine();
                gen.generate(prompt, tokens, temp, topK, topP, System.nanoTime(), System.out::print);
                System.out.print("\n\nStory opening> ");
            }
        }
    }
}
