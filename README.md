# Yoga Feedback - EEG Meditation State Classification
Maulika Patel
Himanshu Mazumdar

**Proof of Knowledge (PoK)** for paper submission / reviewer evaluation.

**Author:** Himanshu S Mazumdar  
**Date:** February 11, 2025

---

## Objective

Simulate EEG-based meditation feedback using Hilbert space-filling curve mapping. This implementation:

- **Classifies** thinking vs. meditating states via spectral band powers (Delta, Theta, Alpha, Beta)
- **Maps** cognitive states to 1D Hilbert coordinates for dimensionality reduction
- **Trains** an MLP neural network classifier
- **Provides** simulated closed-loop feedback for yoga/meditation practice

---

## Repository Contents

| File | Description |
|------|-------------|
| `FeedbackYogaHsm.py` | Main simulation script |
| `eeg_meditation_data.csv` | EEG dataset (channels + label column) |

### Data Format (`eeg_meditation_data.csv`)

- **Columns:** `channel_1` … `channel_512`, `label`
- **Rows:** One EEG epoch per row (2 s at 256 Hz → 512 samples)
- **Labels:** `0` = thinking (high Beta), `1` = meditating (high Alpha/Delta)

---

## Requirements

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

---

## Usage

```bash
python FeedbackYogaHsm.py
```

- Uses `eeg_meditation_data.csv` if present in the same directory.
- Falls back to synthetic data if the CSV is missing.
- Generates 3 plots and prints classification accuracy.

---

## Pipeline Overview

1. **Data loading** — Load CSV or synthetic EEG
2. **Spectral decomposition** — Welch PSD → band powers (Delta, Theta, Alpha, Beta)
3. **Hilbert mapping** — 4D band space → 1D Hilbert coordinate
4. **Classification** — MLP (64, 32) on Hilbert + band features
5. **Visualization** — Hilbert distribution, Alpha vs Beta, trajectory
6. **Feedback** — Simulated closed-loop rules on h(t) and Beta

---

## Contact

**Himanshu S Mazumdar**  
[GitHub @hsmazumdar](https://github.com/hsmazumdar)

