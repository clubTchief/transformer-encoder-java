package com.dictionary;

import com.example.story.CharTokenizer;
import com.tokenizer.BpeTokenizer;
import com.tokenizer.Tokenizer;
import com.transformer.ModelCheckpoint;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Rebuilds a tokenizer from its kind and serialized data (as stored in a checkpoint or in a tokenizer file). */
public final class TokenizerFactory {

    private static final int FILE_MAGIC = 0x544F4B31; // "TOK1"

    private TokenizerFactory() {
    }

    public static Tokenizer load(String kind, byte[] data) {
        switch (kind) {
            case "char":
                return CharTokenizer.deserialize(data);
            case "bpe":
                return BpeTokenizer.deserialize(data);
            default:
                throw new IllegalArgumentException("Unknown tokenizer kind '" + kind + "' (known: char, bpe)");
        }
    }

    public static Tokenizer load(ModelCheckpoint.Loaded checkpoint) {
        return load(checkpoint.tokenizerKind(), checkpoint.tokenizerData());
    }

    /** A tokenizer file lets a trained BPE vocabulary be reused instead of learned again on every run. */
    public static void save(Path file, Tokenizer tokenizer) throws IOException {
        if (file.toAbsolutePath().getParent() != null) {
            Files.createDirectories(file.toAbsolutePath().getParent());
        }
        byte[] data = tokenizer.serialize();
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(file)))) {
            out.writeInt(FILE_MAGIC);
            out.writeUTF(tokenizer.kind());
            out.writeInt(data.length);
            out.write(data);
        }
    }

    public static Tokenizer load(Path file) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(file)))) {
            if (in.readInt() != FILE_MAGIC) {
                throw new IOException(file + " is not a tokenizer file");
            }
            String kind = in.readUTF();
            int length = in.readInt();
            if (length < 0 || length > 200_000_000) {
                throw new IOException("Damaged tokenizer file: " + length + " bytes of data");
            }
            byte[] data = new byte[length];
            in.readFully(data);
            return load(kind, data);
        }
    }
}
