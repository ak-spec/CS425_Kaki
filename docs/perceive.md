# Perceive results

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
