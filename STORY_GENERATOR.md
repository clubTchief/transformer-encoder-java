# Story generator add-on

New package `com.example.story` (no existing file is modified; deps in pom.xml already suffice).

## 1. Get a corpus (TinyStories, ~22 MB validation split is enough for a first run)
```bash
mkdir -p data && curl -L -o data/stories.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```
(The `-train.txt` file is ~2.2 GB; `maxChars=` caps how much of it is read.)

## 2. Train
```bash
mvn -q compile exec:java -Dexec.mainClass=com.example.story.TrainStoryModel \
  -Dexec.args="corpus=data/stories.txt out=checkpoints steps=10000 preset=tiny"
```
Prints train/val loss and a sample story every `evalEvery` steps and checkpoints to `checkpoints/`.

## 3. Generate
```bash
mvn -q exec:java -Dexec.mainClass=com.example.story.GenerateStory -Dexec.args="model=checkpoints temp=0.8"
```

## 4. Tests
`mvn test` -> includes a causality test, an overfit sanity check and sampler tests.

## Variant built on the repo's own DecoderBlock stack
```bash
mvn -q compile exec:java -Dexec.mainClass=com.example.story.TrainBlockStackLM \
  -Dexec.args="corpus=data/stories.txt out=checkpoints-stack steps=10000 accum=8"
mvn -q exec:java -Dexec.mainClass=com.example.story.GenerateStory -Dexec.args="model=checkpoints-stack arch=stack"
```
`BlockStackLM` = learned embeddings -> `DecoderBlock` x N -> final `LayerNorm` -> tied output head.
It processes one sequence per forward pass; `accum` sequences are averaged per optimizer step.
