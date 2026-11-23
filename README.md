# Persian Handwritten Digits: CNN, ResNet and Hybrid Ensemble

CPU-friendly PyTorch pipeline for classifying the **Persian Handwritten Digits** dataset
(GAN-generated, 28×28 grayscale, 150,000 images, 10 classes).

The repository contains two scripts:

| Script | Purpose |
|---|---|
| `persian_digits_cnn.py` | Trains the baseline CNN (single model). |
| `persian_digits_ensemble.py` | Trains a ResNet member and a class-2/3/4 specialist, then evaluates test-time augmentation (TTA), a weighted ensemble and a specialist correction on top of the baseline CNN. |

**Headline result:** the best configuration on both validation and test data is the
**baseline CNN + TTA**, with **99.85% test accuracy (23 errors out of 15,000)**.
The ensemble and the specialist did not improve on it (see [Results](#results) and [Discussion](#discussion)).

---

## Table of contents

1. [Dataset and split](#dataset-and-split)
2. [Methods](#methods)
3. [Installation and usage](#installation-and-usage)
4. [Results](#results)
5. [Discussion](#discussion)
6. [Runtime](#runtime)
7. [Outputs](#outputs)
8. [Notes and limitations](#notes-and-limitations)

---

## Dataset and split

- Source: [Persian Handwritten Digits on Kaggle](https://www.kaggle.com/datasets/amirmahdiabbootalebi/persian-handwritten-digits)
  (`amirmahdiabbootalebi/persian-handwritten-digits`).
- 150,000 images, 28×28, 10 classes (150 × 10 balanced: 15,000 per class).
- Images are read once and cached in `persian_digits_cache.npz`.
- If the mean pixel value is above 127, images are inverted so that the background is 0.
- Stratified split with `SEED = 42`: **80% train (120,000) / 10% validation (15,000) / 10% test (15,000)**.
  The ensemble script reuses exactly the same split, so the test set is identical across both scripts.
- Normalization uses the mean and std of the first 20,000 training images.

> **No test leakage:** every choice in the ensemble script (TTA on/off, model weights, specialist weight α)
> is made on the validation set only. The test set is used for final reporting.

---

## Methods

### 1. Baseline CNN (`persian_digits_cnn.py`)

- Architecture: 6 × (Conv3×3 → BatchNorm → ReLU) with two MaxPool layers (28 → 14 → 7),
  then global average pooling, Dropout(0.3) and a linear layer. Width 32, **288,170 parameters**.
- Training: AdamW (lr 3e-3, weight decay 5e-3), OneCycle schedule, label smoothing 0.05, batch size 256, 12 epochs.
- Augmentation: random affine (rotation ±10°, scale ±10%, translation ±12%).
- EMA of the weights (decay 0.999). The best EMA state by validation accuracy is kept.

### 2. ResNet member (`persian_digits_ensemble.py`)

- Small ResNet: stem conv, then 4 residual blocks (stride 2 at blocks 1 and 3). Width 24, **371,314 parameters**.
- Training: same optimizer, loss and EMA settings, 8 epochs, stronger augmentation (±12°, ±12% scale, ±14% shift).
- Purpose: architectural diversity relative to the baseline CNN.

### 3. Specialist for classes 2, 3, 4

- Same architecture as the baseline CNN, but with a 3-class output (**287,267 parameters**).
- Trained only on the 36,000 training images of classes {2, 3, 4}, 14 epochs, augmentation ±14° / ±14% / ±16%.
- Applied only to samples whose ensemble probability mass on {2, 3, 4} is above 0.5
  (4,501 validation and 4,498 test samples). Its output is blended into the ensemble
  probabilities with weight α (chosen on validation by NLL).

### 4. Test-time augmentation (TTA)

Predictions are averaged over 7 views of each image: the original, ±0.07 horizontal shift,
±0.07 vertical shift and ±6° rotation.

### 5. Ensemble

Weighted average of the baseline CNN and ResNet probabilities. TTA on/off and the weight were selected on
validation NLL, and the result was **TTA on, weight 0.1 (CNN) / 0.9 (ResNet)**.

---

## Installation and usage

```bash
pip install torch numpy pillow
pip install kagglehub matplotlib   # optional: dataset download and plots
```

1. (Optional) set `DATA_DIR` at the top of the scripts to the dataset folder.
   If it is empty or missing, the dataset is downloaded with `kagglehub`.
2. Train the baseline model (creates `persian_digits_cnn.pt` and `persian_digits_cache.npz`):

```bash
python persian_digits_cnn.py
```

3. Run the hybrid pipeline (needs the two files above in the same folder):

```bash
python persian_digits_ensemble.py
```

Trained members (`member_resnet.pt`, `specialist.pt`) are reused if they already exist, so re-running only repeats evaluation.
Delete them to retrain.

---

## Results

### ResNet training (8 epochs)

| Epoch | Loss | Train acc | Val acc (EMA) | Approx. val errors | Time (s) |
|---|---|---|---|---|---|
| 1 | 0.5389 | 93.15% | 99.46% | ~81 | 245 |
| 2 | 0.3241 | 99.19% | 99.66% | ~51 | 314 |
| 3 | 0.3123 | 99.38% | 99.71% | ~43 | 362 |
| 4 | 0.3069 | 99.52% | 99.76% | ~36 | 386 |
| 5 | 0.3034 | 99.60% | 99.77% | ~34 | 382 |
| 6 | 0.3006 | 99.65% | 99.79% | ~31 | 399 |
| 7 | 0.2975 | 99.76% | 99.79% | ~31 | 403 |
| 8 | 0.2962 | 99.78% | 99.79% | ~31 | 392 |

![ResNet training curve](images/resnet_training_curve.png)

Train accuracy is measured with augmentation and dropout, while validation uses the EMA model without augmentation,
which is why validation accuracy is higher than train accuracy during most of training. With label smoothing 0.05 on
10 classes, the theoretical minimum loss is about 0.282, so the final loss of 0.296 is already close to its floor.

### Specialist (classes 2, 3, 4)

Best validation accuracy on its own 3-class task: **99.53%** (epoch 7 of 14, EMA).

### Final comparison on the held-out test set

| Method | Val acc | Test acc | Test errors | Errors in {2, 3, 4} |
|---|---|---|---|---|
| Base CNN | 99.83% | 99.82% | 27 | 20 |
| **Base CNN + TTA** | **99.87%** | **99.85%** | **23** | **16** |
| ResNet | 99.79% | 99.80% | 30 | 22 |
| ResNet + TTA | 99.81% | 99.83% | 26 | 19 |
| Ensemble (TTA on, 0.1 / 0.9) | 99.84% | 99.83% | 26 | 19 |
| Ensemble + Specialist (final pipeline) | 99.84% | 99.83% | 26 | 19 |

The "Base CNN" row reproduces the earlier standalone run exactly (99.82%, 27 errors), confirming the split is identical.

![Test errors per method](images/test_errors_comparison.png)

### Specialist weight α (validation)

| α | Val NLL | Val acc |
|---|---|---|
| **0.0 (chosen)** | **0.04768** | 99.84% |
| 0.1 | 0.04822 | 99.85% |
| 0.2 | 0.04878 | 99.85% |
| 0.3 | 0.04934 | 99.85% |
| 0.4 | 0.04991 | 99.84% |
| 0.5 | 0.05049 | 99.83% |
| 0.6 | 0.05108 | 99.84% |
| 0.7 | 0.05168 | 99.84% |
| 0.8 | 0.05229 | 99.84% |
| 0.9 | 0.05290 | 99.83% |
| 1.0 | 0.05354 | 99.83% |

Validation NLL gets monotonically worse as the specialist weight increases, so α = 0 was selected
(the specialist is effectively switched off).

### Per-class results of the final pipeline

| Class | Accuracy | Errors |
|---|---|---|
| 0 | 100.00% | 0 |
| 1 | 99.93% | 1 |
| 2 | 99.27% | 11 |
| 3 | 99.60% | 6 |
| 4 | 99.87% | 2 |
| 5 | 100.00% | 0 |
| 6 | 99.87% | 2 |
| 7 | 100.00% | 0 |
| 8 | 99.93% | 1 |
| 9 | 99.80% | 3 |

![Final errors per class](images/errors_per_class_final.png)

Most frequent confusions (count: true → predicted): 7: 2→3, 3: 3→4, 3: 3→2, 2: 9→6, 2: 6→9, 2: 4→2, 2: 2→4, 2: 2→1.

---

## Discussion

1. **TTA is the only consistent improvement.** It removed about 4 errors for both models (27 → 23 and 30 → 26),
   at the cost of roughly 7× more inference compute.
2. **The specialist did not help.** Its accuracy on the {2, 3, 4} subtask (99.53% on validation) is no better than that
   of the main models in the same group, and validation NLL worsened for every α > 0, so α = 0 was chosen.
3. **The ensemble did not beat the best single model.** Weights were chosen by validation NLL, which favoured ResNet (0.9),
   while validation *accuracy* favoured the baseline CNN + TTA (99.87% vs. 99.84%). The two criteria disagree here.
   Selecting Base CNN + TTA is also justified on validation data alone, without looking at the test set.
4. **Remaining errors are concentrated in classes 2, 3 and 4** (19 of 26 errors for the final pipeline), mainly 2 → 3.
   Since two different architectures, TTA and a specialist all plateau at roughly 23–30 errors, a share of the remaining
   errors may be ambiguous or noisily labelled GAN-generated samples. Inspect `errors_grid.png` to check this.
5. **Differences are within noise.** With 15,000 test samples, 23 vs. 26 vs. 30 errors is a gap of well under 0.05%,
   comparable to random fluctuation (the standard deviation of an error count near 26 is about 5). Claiming that one method
   is better would require several seeds or a paired test such as McNemar's test (predictions are saved in `ensemble_probs.npz`
   for the ensemble outputs).

**Recommendation:** use the baseline CNN with TTA. It is the simplest strong option and the best on both validation and test.

---

## Runtime

Measured on a CPU-only machine (single run):

| Stage | Time |
|---|---|
| ResNet training (8 epochs) | ~48 min |
| Specialist training (14 epochs) | ~29 min |
| Prediction with TTA (baseline: 171 s, ResNet: 138 s) | ~5 min |
| **Total for the ensemble script** | **83.1 min** |

Epoch times for ResNet grew from 245 s to roughly 390 s with identical workload,
which is most likely CPU throttling or background load rather than a property of the model.

---

## Outputs

| File | Description |
|---|---|
| `persian_digits_cnn.pt` | Baseline CNN checkpoint (state dict, mean, std, width). |
| `member_resnet.pt` | ResNet checkpoint. |
| `specialist.pt` | Specialist checkpoint. |
| `ensemble_log.txt` | Full log of the ensemble run. |
| `ensemble_probs.npz` | Final test and validation probabilities and labels. |
| `errors_grid.png` | Grid of misclassified test images (true / predicted labels). |
| `errors_per_class.png` | Per-class errors, base CNN vs. final pipeline. |

---

## Notes and limitations

- All results come from a **single run with one seed** (`SEED = 42`).
- The dataset is GAN-generated, so results may not transfer to real handwritten digits.
- The specialist gate uses a fixed threshold (0.5 probability mass on {2, 3, 4}); samples the main models assign to other
  classes are never seen by the specialist.
- Hyperparameters were not tuned extensively.

## License

Add a license of your choice here. Check the dataset's own terms on Kaggle before redistributing it.
