# A Step-by-Step Guide to Testing for Quantum Advantage

This document provides a comprehensive framework and a step-by-step experimental process for using the Sequential Forrelation dataset to test for and credibly demonstrate a quantum advantage with your `QuantumMamba` and `QuantumHydra` models.

## 1. The Core Principle: From Query Advantage to Sample Advantage

The Forrelation problem has a proven *query complexity* advantage for quantum computers. In our machine learning context, this theoretical advantage is expected to manifest as a **sample advantage**.

Our central hypothesis is:
> A quantum sequence model can learn the global Forrelation property from significantly fewer examples (i.e., a shorter sequence `L`) than an equally powerful classical model because its quantum state space can represent the underlying Fourier relationship more efficiently.

---

## 2. Step-by-Step Experimental Process

Follow these phases sequentially to build a rigorous case for quantum advantage, from initial baselining to the final scaling analysis.

### Phase 1: Baseline and Hyperparameter Tuning (Prerequisite for All Standards)

**Goal:** Ensure a fair comparison by finding the optimal configuration for *every* model. You cannot claim an advantage if the classical models are not performing at their best.

*   **Step 1.1: Generate a Tuning Dataset**
    Create a single, moderately sized dataset that is challenging but quick to train on. This will be used for all hyperparameter searches.
    ```bash
    python generate_forrelation_dataset.py --num_pairs 2000 --n_bits 6 --seq_len 80 --filename tuning_dataset.pt
    ```

*   **Step 1.2: Perform Hyperparameter Search**
    For **each model** (classical and quantum), perform a systematic search to find its best hyperparameters. Use a tool like Optuna, Ray Tune, or a simple grid search.
    *   **Parameters to Tune:** `learning_rate`, `d_model`, `n_layers`, `d_state`, `expand`, `dropout`.
    *   **Objective:** Maximize the model's validation accuracy on the `tuning_dataset.pt`.

*   **Step 1.3: Record Optimal Configurations**
    For each model, save its best-performing set of hyperparameters. These configurations will be used for all subsequent experiments.

### Phase 2: Testing for the Bronze Standard (Peak Performance)

**Goal:** Compare the best possible performance of the models on a fixed, challenging task.

*   **Step 2.1: Generate a Challenge Dataset**
    Create a new, larger dataset that is more complex than the tuning set.
    ```bash
    python generate_forrelation_dataset.py --num_pairs 5000 --n_bits 7 --seq_len 120 --filename challenge_dataset.pt
    ```

*   **Step 2.2: Train All Models**
    Train each model using its optimal hyperparameters from Phase 1 on the `challenge_dataset.pt`. Ensure the models have a roughly comparable number of trainable parameters.

*   **Step 2.3: Compare Peak Accuracies**
    Compare the final test accuracy of the quantum models against the classical models. If a quantum model achieves a statistically significant higher accuracy, you have met the **Bronze Standard**.

### Phase 3: Testing for the Silver Standard (Sample Efficiency)

**Goal:** Directly measure the "sample advantage" by testing performance on varying sequence lengths.

*   **Step 3.1: Generate a Series of Datasets**
    Create multiple datasets with a fixed complexity (`n_bits`) but a range of sequence lengths (`L`).
    ```bash
    # For L=20
    python generate_forrelation_dataset.py --n_bits 6 --seq_len 20 --filename forrelation_L20.pt
    # For L=40
    python generate_forrelation_dataset.py --n_bits 6 --seq_len 40 --filename forrelation_L40.pt
    # For L=80
    python generate_forrelation_dataset.py --n_bits 6 --seq_len 80 --filename forrelation_L80.pt
    # For L=160
    python generate_forrelation_dataset.py --n_bits 6 --seq_len 160 --filename forrelation_L160.pt
    ```

*   **Step 3.2: Train on All Datasets**
    Train each of your optimized models on each of the generated datasets (`forrelation_L20.pt`, `forrelation_L40.pt`, etc.).

*   **Step 3.3: Plot Accuracy vs. Sequence Length**
    Create a 2D plot where the X-axis is the Sequence Length (`L`) and the Y-axis is the final test accuracy. Plot the learning curve for each model.

*   **Step 3.4: Analyze the Curves**
    Determine the sequence length `L` required for each model to cross a high-accuracy threshold (e.g., 95%). If `L_quantum` is significantly smaller than `L_classical`, you have met the **Silver Standard**.

### Phase 4: Testing for the Gold Standard (Scaling Advantage)

**Goal:** Prove that the quantum advantage becomes more pronounced as the problem gets harder.

*   **Step 4.1: Generate a Harder Series of Datasets**
    Repeat the process from Step 3.1, but increase the problem complexity by using a larger `n_bits`.
    ```bash
    # For n_bits=8, L=40
    python generate_forrelation_dataset.py --n_bits 8 --seq_len 40 --filename forrelation_n8_L40.pt
    # ... repeat for other sequence lengths ...
    ```

*   **Step 4.2: Train on the Harder Datasets**
    Train all your optimized models on this new, more difficult set of datasets.

*   **Step 4.3: Compare the Learning Curve Gaps**
    Create two plots (or a single plot with faceted panels): one showing the "Accuracy vs. `L`" curves for `n_bits=6`, and one for `n_bits=8`.

*   **Step 4.4: Analyze the Scaling**
    Compare the *gap* in sample efficiency between the quantum and classical models on both plots. If the performance gap *widens* on the harder (`n_bits=8`) dataset, it demonstrates a superior scaling advantage. Achieving this, in addition to the Silver Standard, meets the **Gold Standard** of quantum advantage.

---

## 3. Standards and Thresholds: A Pyramid of Evidence

This is a summary of the claims you can make based on the experimental phases.

| Standard | Primary Metric | Threshold for Advantage | Strength of Claim |
| :--- | :--- | :--- | :--- |
| **Bronze** | Peak Accuracy | Quantum > Classical (statistically significant) on a fixed, hard task. | Suggestive |
| **Silver** | **Sample Efficiency** | `L_quantum` << `L_classical` to achieve the same high accuracy. | **Strong** |
| **Gold** | **Scaling & Efficiency** | Silver Standard is met, AND the performance gap widens as `n_bits` increases. | **Compelling Proof** |
