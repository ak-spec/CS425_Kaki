# Perceive

Perceive predicts the seeker's emotion and problem type from the conversation so far. It is trained in
`train_perceive.ipynb`; `src/perceive.py` loads the checkpoint for the pipeline.

## Training

Open `train_perceive.ipynb` and run all cells (set `smoke_test=True` in `CFG` for a quick pipeline check on a small subset).

- **Model**: one shared DistilRoBERTa encoder (6 layers, 12 attention heads, hidden size 768) feeding two linear classifier heads, one for emotion and one for problem type, both reading the `<s>` token.
- **Input**: the `text` field, truncated from the **left** to 512 tokens so the latest seeker message always survives.
- **Loss**: cross-entropy per head, summed, with tempered class weights computed from the train split only.
- **Model selection**: best epoch by the mean of the two validation macro-F1 scores. The test split is scored once, at the end.

### Hyper-parameters

| Parameter | Value |
|---|---|
| Base model | `distilroberta-base` |
| Max length | 512 tokens (left truncation) |
| Batch size | 16 |
| Optimiser | AdamW, learning rate 2e-5, weight decay 0.01 |
| Schedule | linear decay, 10% warmup |
| Epochs | 4 |
| Dropout (before heads) | 0.1 |
| Gradient clipping | 1.0 |
| Class weights | (N / (K * n_c)) ** 0.5, train split only |
| Precision | fp16 autocast on CUDA, float32 otherwise |
| Seed | 42 |

## What the notebook reports

The notebook reports, for validation and test:

- accuracy, macro-F1 and weighted-F1 for each head, overall and on the in-scope subset
- accuracy by seeker-turn position (turns 1, 2, 3, 4-5, 6-8, 9+), because early histories often contain too little evidence to name the emotion
- per-class precision, recall and F1 on the test split
- a check of whether averaging class probabilities over the turns so far improves predictions

It saves `checkpoints/perceive/best.pt`, `run.json` (configuration, per-epoch history and reports) and `perceive_preds_{val,test}.jsonl` (predicted probabilities per sample, for analysis). The pipeline runs Perceive live through `src/perceive.py`, so Speak gets the predicted emotion, not the reference one.

Macro-F1 is computed over the classes that occur in the evaluated split, so classes with no examples there (for example emotion `other` in val and test) are not scored. Some classes have very few validation and test dialogues, so their per-class numbers are noisy.

### Platforms

`train_perceive.ipynb` runs locally, on Kaggle and on Google Colab from `main`. Its first code cell detects the platform and sets the data and output paths; the note under the title has the steps for each platform.

## Results

Full training run of `train_perceive.ipynb` with its default `CFG` (distilroberta-base, 512 tokens,
batch 16, lr 2e-5, 4 epochs, seed 42), on one RTX 3090 (about 70 seconds per epoch). The splits
fingerprint was `dd619339a81a`. Epoch 2 had the best mean validation macro-F1 and was kept; the test
split was scored once with that checkpoint.

| Split | Emotion acc | Emotion macro-F1 | Problem acc | Problem macro-F1 |
|---|---|---|---|---|
| val | 0.541 | 0.376 | 0.771 | 0.715 |
| test | 0.536 | 0.414 | 0.750 | 0.670 |
| test, in-scope problems | 0.551 | 0.430 | 0.788 | 0.801 |

Macro-F1 covers the classes present in each split (emotion `other` does not occur in val or test).

Accuracy by seeker-turn position on test (emotion / problem): turn 1 0.43 / 0.45, turn 2 0.51 / 0.67,
turn 3 0.57 / 0.81, turns 4-5 0.55 / 0.80, turns 6-8 0.57 / 0.82, turns 9+ 0.54 / 0.76. The first
message alone rarely says enough to name the problem.

Averaging the class probabilities over the seeker turns so far raises test accuracy from 0.536 to 0.546
(emotion) and from 0.750 to 0.768 (problem). This only changes inference.

Weakest classes on test: emotion `shame` (F1 0.05, 20 samples), `disgust` (0.26) and `fear` (0.40);
problem `other` (0.29) and `conflict with parents` (0.42, 33 samples). These are the smallest classes.

## Inference

`src/perceive.py` loads the notebook's checkpoint (`checkpoints/perceive/best.pt` and `run.json`) with
the label names from `data/processed/label_maps.json`:

```python
from perceive import Perceiver
perceiver = Perceiver("checkpoints/perceive")
perceiver.predict("supporter: Hi, how are you? | seeker: I failed my midterm.")
# {"emotion": ..., "emotion_probs": {...}, "problem": ..., "problem_probs": {...}}
```

On 40 test histories it reproduces the notebook's saved predictions exactly. A prediction takes about
30 ms on a laptop CPU.
