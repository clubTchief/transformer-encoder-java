# Training the story generator on a text file

`CorpusTrainer` trains the repo's own decoder stack (`BlockStackLM`: embeddings, `DecoderBlock`s with RoPE, final
RMSNorm, tied output head) on any text file, one character at a time. It writes a **self-describing checkpoint**
(`ModelCheckpoint`): architecture, vocabulary, training window and weights are in one file, so the generator needs
no hard-coded corpus or sizes. `CorpusGeneratorCLI` loads that file and continues whatever you type.

## 1. Get a corpus (Windows `cmd`)
TinyStories' validation split is ~22 MB, enough for a first run:
```
mkdir data
curl -L -o data\stories.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```
Stories in that file are separated by `<|endoftext|>`; the tokenizer turns the marker into the end-of-story token, which
is how the model learns to finish a story. Any other UTF-8 text works as well.

## 2. Train
```
mvn -q compile exec:java -Dexec.mainClass=com.dictionary.CorpusTrainer -Dexec.args="corpus=data/stories.txt steps=3000"
```
It prints the training loss every 50 steps and, every `evalEvery` steps, the validation loss and a sample story, and it
saves `models/tinystories.ckpt` each time (the file is replaced atomically, so you can stop training at any moment).

| argument | default | meaning |
| --- | --- | --- |
| `corpus` | (required) | text file |
| `out` | `models/tinystories.ckpt` | checkpoint to write |
| `steps` | 5000 | optimizer steps |
| `batch`, `seqLen` | 32, 128 | windows per step, characters per window (= context the model is trained on) |
| `layers`, `dModel`, `heads`, `dFF` | 4, 128, 4, 352 | architecture |
| `lr`, `wd`, `clip`, `warmup` | 1e-3, 0.1, 1.0, auto | peak learning rate (cosine decay), weight decay, gradient-norm clip, warm-up steps (auto: max(100, steps/20)) |
| `init` | 0.02 | GPT-style initialisation: normal(0, init) weights, and init / sqrt(2 * layers) for the matrices that write into the residual stream; `init=0` selects the legacy uniform initialisation |
| `logEvery` | 50 | print the average loss every N steps (the first 10 steps are always printed one by one) |
| `evalEvery`, `evalBatches`, `sampleTokens` | 500, 8, 200 | evaluation cadence and sample length (0 = no sample) |
| `maxChars`, `minFreq`, `valFraction`, `seed` | 20000000, 50, 0.01, 1337 | how much text to read, rare characters become `?`, validation share, RNG seed |

A misspelled argument is an error. Larger models (`layers=6 dModel=256 heads=8 dFF=704 seqLen=256`) give better
text and need a lot more time on a CPU.

## 3. Generate
```
mvn -q exec:java -Dexec.mainClass=com.dictionary.CorpusGeneratorCLI -Dexec.args="model=models/tinystories.ckpt temp=0.8"
```
Type the start of a story (an empty line starts a fresh one), `exit` to quit. Options: `tokens` (400), `temp` (0.8,
0 = greedy), `topk` (40), `topp` (0.95), `seed`, and `prompt=...` for a single non-interactive run.

Generation uses the KV cache with a **sliding window**: when the cache is full, the newest half is kept and rebuilt from
position 0, so a story can be longer than `seqLen` without leaving the positions the model was trained on.

## Why this replaces the word-level demo for real text
`StoryGeneratorCLI` and `MultiBatchTrainer` use a closed word vocabulary built from four hard-coded sentences, so
they cannot even read "galaxy". The character model has no unknown words, trains on random windows of the whole corpus
(not fixed chunks that cut sentences apart), and its checkpoint cannot drift from the code that reads it.

## If training reports `The loss became NaN`
The trainer never applies an update after a non-finite loss, and it skips (and counts) steps whose gradient is NaN or
Infinity instead of letting them turn every weight into NaN. When it does stop, the message lists the last losses and
the tensors (`layer2.wG`, `embedding`, ...) whose weights or gradient are not finite. "All weights and gradients are
finite" means the overflow happened inside the forward pass of that batch. Things to try, in this order:
1. a smaller learning rate, e.g. `lr=5e-4`;
2. a longer warm-up, e.g. `warmup=300`;
3. `logEvery=1` to see the loss and the gradient norm at every step (a norm that keeps growing is the warning sign).
Please keep the printed message when reporting a problem; it says where the numbers went wrong.
