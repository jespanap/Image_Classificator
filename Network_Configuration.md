# Network Configuration Document — DL Competition 01

## Architecture Description

**Model: DeepMLP** — A deep Multi-Layer Perceptron with Batch Normalization and Dropout for 6-class image classification. No convolutional layers are used.

### Architecture Overview

```
Input: 1 × 150 × 150 = 22,500 (flattened grayscale image)
  → Layer 1: Linear(22,500 → 1,024) + BN + ReLU + Dropout(0.4)
  → Layer 2: Linear(1,024  → 512)   + BN + ReLU + Dropout(0.4)
  → Layer 3: Linear(512    → 256)   + BN + ReLU + Dropout(0.3)
  → Layer 4: Linear(256    → 128)   + BN + ReLU + Dropout(0.2)
  → Output:  Linear(128 → 6)        — class logits
```

### Detailed Layer Breakdown

#### Hidden Layers

| Layer   | Input Dim | Output Dim | Activation | BatchNorm | Dropout |
|---------|-----------|------------|------------|-----------|---------|
| Layer 1 | 22,500    | 1,024      | ReLU       | Yes       | 0.4     |
| Layer 2 | 1,024     | 512        | ReLU       | Yes       | 0.4     |
| Layer 3 | 512       | 256        | ReLU       | Yes       | 0.3     |
| Layer 4 | 256       | 128        | ReLU       | Yes       | 0.2     |
| Output  | 128       | 6          | —          | No        | —       |

### Total Parameters

Approximately **24M** parameters. The first linear layer dominates (22,500 × 1,024 ≈ 23M), which is inherent to MLP architectures operating on flattened images.

### Weight Initialization

- **Linear layers:** Kaiming Normal (He initialization), biases set to zero
- **BatchNorm layers:** weight = 1, bias = 0

---

## Input Size and Preprocessing

- **Input size:** 1 × 150 × 150 (grayscale), flattened to a 22,500-dimensional vector
- **Preprocessing pipeline (applied to all splits equally):**
  - `Grayscale(num_output_channels=1)` — single channel, reduces input dimensionality
  - `Resize((150, 150))` — fixed spatial resolution
  - `ToTensor()` — converts to tensor, values in [0, 1]
  - `Normalize(mean=[0.5], std=[0.5])` — maps pixel values to approximately [−1, 1]
- No data augmentation is applied. The same transform is used for train, val, test, and comp_test to ensure consistency.

---

## Loss Function

**CrossEntropyLoss** with **label smoothing = 0.1** — softens one-hot targets to reduce overconfidence and improve generalization. For the correct class, the target becomes 0.9167 instead of 1.0; for incorrect classes, 0.0167 instead of 0.

---

## Optimizer and Hyperparameters

| Parameter     | Value                              |
|---------------|------------------------------------|
| Optimizer     | Adam                               |
| Learning rate | 0.001                              |
| Weight decay  | 1e-4                               |
| Batch size    | 32                                 |
| Max epochs    | 40                                 |
| Scheduler     | StepLR (step\_size=10, gamma=0.5)  |
| Gradient clip | max\_norm = 1.0                    |

### Scheduler Details

StepLR reduces the learning rate by a factor of 0.5 every 10 epochs:

| Epochs    | Learning Rate |
|-----------|---------------|
| 1 – 10    | 1e-3          |
| 11 – 20   | 5e-4          |
| 21 – 30   | 2.5e-4        |
| 31 – 40   | 1.25e-4       |

This allows large steps early in training for fast convergence, then finer updates in later epochs for better optimization.

---

## Regularization Methods

1. **Dropout (p=0.2 to 0.4):** Applied after every hidden layer with decreasing rates in deeper layers:
   - Layer 1: 0.4
   - Layer 2: 0.4
   - Layer 3: 0.3
   - Layer 4: 0.2

2. **Batch Normalization (BatchNorm1d):** Applied after every linear layer before the activation. Normalizes activations within each batch, which stabilizes training, reduces sensitivity to initialization, and acts as implicit regularization.

3. **Weight Decay (L2 Regularization):** 1e-4 via Adam's weight decay parameter. Penalizes large weights and encourages the model to find simpler solutions.

4. **Label Smoothing (0.1):** Prevents the model from becoming overconfident on training examples, which tends to improve calibration and generalization to unseen data.

5. **Gradient Clipping (max_norm=1.0):** Clips the global gradient norm to 1.0 before each optimizer step. Prevents gradient explosions, which are common in deep MLPs with large input dimensions.

6. **Early Stopping (patience=8):** Training halts if validation accuracy does not improve for 8 consecutive epochs. The best model weights are restored afterwards.

---

## Design Rationale — MLP Approach

The MLP approach flattens the image into a vector, treating each pixel as an independent feature. This introduces specific challenges and design decisions:

- **No spatial hierarchy:** Unlike CNNs, the MLP does not learn local patterns through kernels and pooling. All spatial relationships must be learned purely from data, requiring strong regularization.

- **Grayscale over RGB:** The competition baseline uses a single grayscale channel, which reduces the input from 67,500 to 22,500 features. This significantly decreases the parameter count and training time, at the cost of losing color information.

- **BatchNorm at every layer:** Without BatchNorm, deep MLPs on high-dimensional inputs tend to suffer from internal covariate shift, slow convergence, and sensitivity to the learning rate. BatchNorm mitigates all three.

- **ReLU over GELU:** ReLU is simpler, faster to compute, and well-understood. It performs comparably to GELU for standard MLP architectures with BatchNorm, since BatchNorm already controls the activation distribution entering each layer.

- **Progressive dimensionality reduction:** The architecture halves the hidden dimension at each step (1024 → 512 → 256 → 128). This forces the model to compress information gradually, learning increasingly abstract representations.

---

## Model Selection Strategy

- The model is evaluated on a **stratified validation sample (100 images)** after every epoch, drawn from seg_test using `StratifiedShuffleSplit` to maintain class balance.
- The model weights achieving the **highest validation accuracy** are saved to `best_model.pth` using `copy.deepcopy` of the state dict.
- **Early stopping** with patience=8 halts training if no improvement is observed.
- These best weights are loaded before generating competition predictions on comp_test.
- A final sanity check is run on a separate **test sample (100 images)** — also from seg_test but drawn independently — to estimate generalization performance.
- Per-class accuracy is reported on the test sample to identify potential weaknesses across the 6 scene classes.