package com.transformer;

import ai.djl.ndarray.NDManager;
import com.tokenizer.Tokenizer;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Self-describing checkpoint: one file holding everything needed to use a trained model again, so a generator does
 * not have to repeat (and possibly get wrong) the hyper-parameters, the tokenizer or the corpus it was trained on.
 *
 * <pre>
 * int    MAGIC                      0x534C4D43 ("SLMC")
 * int    VERSION                    2 (version 1, which stored a bare character vocabulary, is still read)
 * int    numLayers, dModel, numHeads, dFF, vocabSize;  float eps       (LMConfig)
 * UTF    tokenizer kind             "char" or "bpe"
 * int    tokenizer data length, then that many bytes                   (Tokenizer.serialize())
 * int    metadata entries, then UTF key / UTF value pairs              (seqLen, steps, validation loss, ...)
 * ...    the tensors, as written by ModelSerializer.writeTensors
 * </pre>
 */
public final class ModelCheckpoint {

    public static final int MAGIC = 0x534C4D43;
    public static final int VERSION = 2;

    /** A loaded checkpoint: the model with its weights, the tokenizer's kind and data, and the metadata. */
    public record Loaded(BlockStackLM model, String tokenizerKind, byte[] tokenizerData, Map<String, String> metadata) {

        /** The vocabulary of a "char" checkpoint (the characters of the ids 0, 1, 2, ...). */
        public char[] vocabulary() {
            if (!"char".equals(tokenizerKind)) {
                throw new IllegalStateException("This checkpoint uses a '" + tokenizerKind + "' tokenizer, not a character vocabulary");
            }
            char[] chars = new char[tokenizerData.length / 2];
            for (int i = 0; i < chars.length; i++) {
                chars[i] = (char) (((tokenizerData[2 * i] & 0xFF) << 8) | (tokenizerData[2 * i + 1] & 0xFF));
            }
            return chars;
        }
    }

    private ModelCheckpoint() {
    }

    public static void save(Path path, BlockStackLM model, Tokenizer tokenizer, Map<String, String> metadata) throws IOException {
        if (tokenizer.vocabSize() != model.getConfig().vocabSize()) {
            throw new IllegalArgumentException("The tokenizer has " + tokenizer.vocabSize() + " tokens but the model has vocabSize "
                    + model.getConfig().vocabSize());
        }
        write(path, model, tokenizer.kind(), tokenizer.serialize(), metadata);
    }

    /** Character vocabulary variant (kept for callers that have only the characters). */
    public static void save(Path path, BlockStackLM model, char[] vocabulary, Map<String, String> metadata) throws IOException {
        if (vocabulary.length != model.getConfig().vocabSize()) {
            throw new IllegalArgumentException("Vocabulary has " + vocabulary.length + " entries but the model has vocabSize "
                    + model.getConfig().vocabSize());
        }
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (DataOutputStream out = new DataOutputStream(bytes)) {
            for (char c : vocabulary) {
                out.writeChar(c);
            }
        }
        write(path, model, "char", bytes.toByteArray(), metadata);
    }

    /** Writes to a temporary file first and then moves it into place, so an interrupted save never destroys a good file. */
    private static void write(Path path, BlockStackLM model, String tokenizerKind, byte[] tokenizerData,
                              Map<String, String> metadata) throws IOException {
        LMConfig config = model.getConfig();
        if (path.toAbsolutePath().getParent() != null) {
            Files.createDirectories(path.toAbsolutePath().getParent());
        }
        Path tmp = path.resolveSibling(path.getFileName() + ".tmp");

        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(tmp)))) {
            out.writeInt(MAGIC);
            out.writeInt(VERSION);
            out.writeInt(config.numLayers());
            out.writeInt(config.dModel());
            out.writeInt(config.numHeads());
            out.writeInt(config.dFF());
            out.writeInt(config.vocabSize());
            out.writeFloat(config.eps());

            out.writeUTF(tokenizerKind);
            out.writeInt(tokenizerData.length);
            out.write(tokenizerData);

            out.writeInt(metadata.size());
            for (Map.Entry<String, String> entry : metadata.entrySet()) {
                out.writeUTF(entry.getKey());
                out.writeUTF(entry.getValue());
            }

            ModelSerializer.writeTensors(out, model.params());
        }

        try {
            Files.move(tmp, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(tmp, path, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /** Builds the model from the stored architecture and fills it with the stored weights. */
    public static Loaded load(NDManager manager, Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            int magic = in.readInt();
            if (magic != MAGIC) {
                throw new IOException(path + " is not a self-describing story-LM checkpoint "
                        + "(a bare weights file written by ModelSerializer.saveWeights has no header)");
            }
            int version = in.readInt();
            if (version != 1 && version != VERSION) {
                throw new IOException("Unsupported checkpoint version " + version + " (this build reads versions 1 and " + VERSION + ")");
            }

            int numLayers = in.readInt();
            int dModel = in.readInt();
            int numHeads = in.readInt();
            int dFF = in.readInt();
            int vocabSize = in.readInt();
            float eps = in.readFloat();
            LMConfig config;
            try {
                config = new LMConfig(numLayers, dModel, numHeads, dFF, vocabSize, eps);
            } catch (IllegalArgumentException e) {
                throw new IOException("Damaged checkpoint header: " + e.getMessage(), e);
            }

            String tokenizerKind;
            byte[] tokenizerData;
            if (version == 1) {
                // version 1 stored the characters of a character vocabulary
                int vocabLength = in.readInt();
                if (vocabLength != vocabSize) {
                    throw new IOException("Damaged checkpoint: vocabulary has " + vocabLength + " entries but vocabSize is " + vocabSize);
                }
                ByteArrayOutputStream bytes = new ByteArrayOutputStream();
                DataOutputStream chars = new DataOutputStream(bytes);
                for (int i = 0; i < vocabLength; i++) {
                    chars.writeChar(in.readChar());
                }
                tokenizerKind = "char";
                tokenizerData = bytes.toByteArray();
            } else {
                tokenizerKind = in.readUTF();
                int length = in.readInt();
                if (length < 0 || length > 200_000_000) {
                    throw new IOException("Damaged checkpoint: tokenizer data of " + length + " bytes");
                }
                tokenizerData = new byte[length];
                in.readFully(tokenizerData);
            }
            if ("char".equals(tokenizerKind) && tokenizerData.length / 2 != vocabSize) {
                throw new IOException("Damaged checkpoint: the character vocabulary has " + tokenizerData.length / 2
                        + " entries but vocabSize is " + vocabSize);
            }

            int entries = in.readInt();
            if (entries < 0 || entries > 10_000) {
                throw new IOException("Damaged checkpoint: " + entries + " metadata entries");
            }
            Map<String, String> metadata = new LinkedHashMap<>();
            for (int i = 0; i < entries; i++) {
                metadata.put(in.readUTF(), in.readUTF());
            }

            BlockStackLM model = new BlockStackLM(manager, config);
            try {
                ModelSerializer.readTensors(in, model.params());
            } catch (EOFException e) {
                throw new IOException("Damaged checkpoint: the file ends before all tensors were read", e);
            }
            return new Loaded(model, tokenizerKind, tokenizerData, metadata);
        }
    }

    /** True if the file starts with this format's magic number (cheap check, reads 4 bytes). */
    public static boolean isCheckpoint(Path path) {
        try (DataInputStream in = new DataInputStream(Files.newInputStream(path))) {
            return in.readInt() == MAGIC;
        } catch (IOException e) {
            return false;
        }
    }
}
