# PMU Source Identification using GPT4TS

A deep learning framework for identifying the source of synchrophasor (PMU) measurements using foundation models for time-series classification.

The project investigates whether pretrained transformer architectures can learn unique fingerprints of Phasor Measurement Units (PMUs) from high-frequency synchrophasor data collected from the Victorian electricity distribution network.

---

## Overview

Phasor Measurement Units (PMUs) continuously stream synchronized voltage, current and frequency measurements used for monitoring and controlling modern power systems.

This project explores **PMU source identification**, where the goal is to determine **which PMU generated a given sequence of measurements**. The problem is formulated as a multi-class time-series classification task involving approximately 50 PMUs.

Unlike traditional fingerprinting approaches that rely on handcrafted statistical features, this work investigates adapting pretrained language-model architectures to multivariate time-series data using the **GPT4TS** framework.

Current experiments focus on:

- GPT4TS for multivariate time-series classification
- Transfer learning from pretrained GPT-2
- Patch-based time-series tokenization
- PMU fingerprinting using frequency and voltage phasors
- Evaluation using accuracy, macro F1-score and confusion matrices

---

## Repository Structure

```text
power-anomaly-detection/
│
├── data/                       # PMU CSV files
├── checkpoints/                # Saved model checkpoints
│
├── src/
│   ├── datasets/
│   │   ├── PMUData.py          # Data loading and preprocessing
|   |   └── dataset.py          # Dataset used by pytorch
│   │
│   ├── models/
│   │   ├── gpt4ts.py           # GPT4TS classifier
│   │   └── embed.py            # Patch embedding layers
│   │
│   └── visualise.py            # visualise training/validation progress
│
├── main.py                     # Training / validation pipeline
├── script.sh                   # Example training configuration
├── visualise.sh                # Example visualisation script
├── README.md
└── requirements.txt
```

---

# Dataset

The dataset consists of synchronized PMU measurements collected from the Victorian electricity distribution network.

### Characteristics

| Property | Value |
|----------|-------|
| PMUs | ~49 |
| Sampling rate | 50 Hz |
| Sample interval | 20 ms |
| Duration | Three one-hour recordings per PMU |
| Format | CSV |

Each CSV contains synchronized measurements including:

- Frequency
- GPS status flag
- Voltage magnitude and angle
- Current magnitude and angle

Example input features include

```
FREQ
UA:MAG
UA:ANG
UB:MAG
UB:ANG
UC:MAG
UC:ANG
```

Additional channels can easily be incorporated through the command-line interface.

---

# Data Preprocessing

Each PMU is preprocessed independently before training.

The preprocessing pipeline performs:

1. Duplicate timestamp removal
2. Timestamp parsing
3. Uniform 20 ms reindexing
4. Missing value interpolation (small gaps only)
5. GPS lock validation using the FLAG field
6. Removal of invalid sections
7. Continuous segment identification
8. Sliding-window generation

Segment IDs ensure that no training window crosses a discontinuity introduced by missing data or GPS lock loss.

---

# Model

The classifier is based on **GPT4TS**, which adapts a pretrained GPT-2 model for multivariate time-series classification.

The pipeline consists of

```
Input Window
      │
      ▼
Patch Extraction
      │
      ▼
Patch Embedding
      │
      ▼
Pretrained GPT-2 Backbone
(first six transformer blocks)
      │
      ▼
LayerNorm
      │
      ▼
Dropout
      │
      ▼
Linear Classification Head
      │
      ▼
Predicted PMU
```

The GPT-2 backbone is initialized from pretrained weights, while the classification head is trained for PMU identification.

---

# Training

Training uses:

- Cross-entropy loss
- Adam optimizer
- Separate learning rates for backbone and classifier
- Early stopping
- Checkpointing
- Validation after every epoch
- Optional GPU acceleration

The training script records

- Training accuracy
- Training loss
- Validation accuracy
- Validation macro F1-score
- Validation loss
- Runtime
- Memory usage

---

# Running Experiments

Example configuration:

```bash
python3 main.py \
    --data-dir ./data \
    --columns FREQ UA:MAG UA:ANG UB:MAG UB:ANG UC:MAG UC:ANG \
    --window-length 500 \
    --stride 400 \
    --batch-size 32 \
    --epochs-per-run 100 \
    --head-learning-rate 3e-4 \
    --backbone-learning-rate 1e-5 \
    --dropout 0.1 \
    --patch-size 10 \
    --patch-stride 10 \
    --d-model 768 \
    --patience 15 \
    --l2-lambda 0.001
```

---

# Important Hyperparameters

| Parameter | Description |
|-----------|-------------|
| window-length | Length of each input sequence |
| stride | Sliding-window stride |
| patch-size | Number of samples per patch |
| patch-stride | Distance between adjacent patches |
| d-model | GPT embedding dimension |
| head-learning-rate | Learning rate of classification head |
| backbone-learning-rate | Learning rate of pretrained GPT layers |
| dropout | Dropout before classification |
| l2-lambda | L2 regularization strength |
| patience | Early stopping patience |

---

# Evaluation

Performance is evaluated using

- Classification accuracy
- Macro F1-score
- Cross-entropy loss
- Confusion matrix

The framework supports experiments on varying numbers of PMUs, allowing evaluation of scalability from small subsets to the complete 49-class identification task.

---

# Current Research Focus

This project investigates:

- Can pretrained language models identify PMUs directly from raw synchrophasor measurements?
- How well do pretrained representations transfer to power-system time-series?
- How does classification performance scale with increasing numbers of PMUs?
- Which combinations of electrical measurements provide the strongest PMU fingerprints?

---

# Requirements

- Python 3.11
- PyTorch
- Transformers
- pandas
- numpy
- scikit-learn
- einops
- matplotlib

Install dependencies with

```bash
pip install -r requirements.txt
```

---

# Future Work

Possible extensions include

- Fine-tuning additional GPT layers
- Comparison against PatchTST and iTransformer
- Self-supervised pretraining on PMU data
- Robustness against noisy or spoofed measurements
- Cross-network generalization
- Explainability of learned PMU fingerprints

---

# Author

**Ethan Harvey**

Undergraduate Research Project

School of Electrical Engineering and Computer Science

The University of Queensland