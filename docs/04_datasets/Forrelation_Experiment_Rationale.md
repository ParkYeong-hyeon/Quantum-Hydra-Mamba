# Rationale for Using Forrelation as a BQP-Complete Benchmark

This document details the motivation and experimental design for using the **Forrelation** problem as a BQP-complete task to validate and benchmark the Quantum Mamba and Quantum Hydra models.

## 1. The Forrelation Problem

### Mathematical Definition

The Forrelation problem, introduced by Aaronson & Ambainis (2015), asks whether two Boolean functions f, g : {0,1}^n -> {-1,+1} have high or low "forrelation" value:

```
Phi(f,g) = (1 / 2^{3n/2}) * SUM_{x,y in {0,1}^n} f(x) * (-1)^{x.y} * g(y)
```

This is equivalent to measuring the correlation between f and the Walsh-Hadamard (Fourier) transform of g.

### Classification Task

The problem is to decide:
- **Forrelated (label=1)**: Phi(f,g) >= 3/5 (approximately 0.6)
- **Unforrelated (label=0)**: |Phi(f,g)| <= 1/100 (approximately 0)

## 2. Why Forrelation is Ideal for Quantum Advantage Testing

### 2.1 Proven Maximal Quantum-Classical Separation

The Aaronson & Ambainis (2015) paper proves that:

| Algorithm Type | Query Complexity |
|---------------|------------------|
| **Quantum** | O(1) - single query |
| **Classical Randomized** | Omega(sqrt(N) / log(N)) |

This is the **largest possible** separation between quantum and classical query complexity for decision problems.

### 2.2 BQP-Completeness

The k-fold generalization of Forrelation is **BQP-complete** (Theorem 5 in the paper). This means:
- Any problem solvable by a quantum computer can be reduced to Forrelation
- Forrelation "captures the full power of quantum computation"
- Solving Forrelation efficiently implies the ability to simulate quantum circuits

### 2.3 Tests Fourier Transform Learning

The Forrelation problem is fundamentally about detecting hidden correlations in the Fourier domain:
- A quantum algorithm uses the Hadamard transform to detect this correlation instantly
- A classical algorithm must laboriously query many function values to statistically infer the correlation
- This directly tests whether quantum-inspired models can learn Fourier-like relationships

## 3. Dataset Generation: The Gaussian Rounding Method (V2)

### 3.1 The Bug in V1 (Fixed)

The original V1 implementation had a critical **data leakage bug**:
- High forrelation: f was a delta function (99% values = -1, one value = +1)
- Low forrelation: f was random (50% each)

Classical models could achieve 100% accuracy by simply counting the distribution of f values without computing any Forrelation!

### 3.2 Correct Implementation (V2) - Based on Theorem 6

The V2 implementation uses the **Gaussian rounding method** from Theorem 6 of the paper:

**For FORRELATED pairs (label=1):**
1. Generate f_real ~ N(0,1)^N (standard Gaussian for each entry)
2. Compute g_real = Walsh-Hadamard(f_real) / sqrt(N)
3. Round to Boolean: f = sign(f_real), g = sign(g_real)
4. Expected forrelation: E[Phi(f,g)] = 2/pi ~ 0.637

**For UNFORRELATED pairs (label=0):**
1. Generate f_real ~ N(0,1)^N independently
2. Generate g_real ~ N(0,1)^N independently
3. Round to Boolean: f = sign(f_real), g = sign(g_real)
4. Expected forrelation: E[|Phi(f,g)|] ~ 0

### 3.3 Why This is BQP-Complete

In BOTH classes:
- f looks like a random Boolean function (~50% positive values)
- g looks like a random Boolean function (~50% positive values)
- Individual statistics of f and g are **indistinguishable** between classes

The ONLY difference is the **hidden correlation structure**:
- In forrelated pairs: g is correlated with the Fourier transform of f
- In unforrelated pairs: g is independent of f

This makes it **impossible** for classical models to distinguish classes by looking at statistical properties. They must actually compute (or learn to approximate) the Forrelation!

### 3.4 Verified Statistics (V2)

```
Forrelated pairs (label=1):
  Mean forrelation: 0.638 (expected: 2/pi = 0.637)
  Mean f positive fraction: 0.500 (expected: 0.50)

Unforrelated pairs (label=0):
  Mean |forrelation|: 0.065 (expected: ~0)
  Mean f positive fraction: 0.500 (expected: 0.50)

Data leakage check: PASSED (difference < 0.001)
Class separation: GOOD (min forrelated > max |unforrelated|)
```

## 4. The Sequential Forrelation Task

### 4.1 Sequence Generation

For each function pair (f, g), we generate a sequence of L samples:

**Input at timestep t:**
```
u[t] = [x_bits, f(x), y_bits, g(y)]
```
where:
- x_bits: binary representation of randomly sampled x
- f(x): the function value at x in {-1, +1}
- y_bits: binary representation of randomly sampled y
- g(y): the function value at y in {-1, +1}

**Output:** Binary classification (forrelated vs unforrelated)

### 4.2 The Challenge

The model must learn to:
1. Process a sequence of random samples from f and g
2. Detect the hidden Fourier correlation between f and g
3. Classify whether Phi(f,g) is high or low

This requires learning a **global property** from **local observations** - the exact scenario where quantum algorithms excel.

## 5. Experimental Hypotheses

### Primary Hypothesis
Quantum-inspired models (Groups 1, 2, 4) will show **better performance** than classical baselines (Group 3) on this task because:
- The task directly probes Fourier-domain correlations
- Quantum circuits naturally implement the Walsh-Hadamard transform
- Classical models lack efficient mechanisms to detect these correlations

### Secondary Hypotheses

1. **Sample Efficiency**: Quantum models may achieve high accuracy with shorter sequence lengths
2. **Parameter Efficiency**: Quantum models may achieve similar accuracy with fewer parameters
3. **Scaling**: Performance gap may widen as n_bits increases (larger N = 2^n)

## 6. Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| **Test Accuracy** | Primary measure of classification performance |
| **Test AUC** | Robustness across decision thresholds |
| **Test F1** | Balance between precision and recall |
| **Baseline** | Random chance (0.50 for balanced classes) |

## 7. Expected Results

If quantum-inspired architectures have genuine advantages:
- Groups 1, 2, 4 should achieve accuracy significantly above baseline
- Group 3 (classical) should struggle, with accuracy near baseline (~50%)

If there is no quantum advantage:
- All groups should perform similarly
- This would suggest the quantum components don't help with Fourier correlation detection

## 8. References

1. Aaronson, S., & Ambainis, A. (2015). "Forrelation: A Problem that Optimally Separates Quantum from Classical Computing." STOC'15.
   - Full paper: https://www.scottaaronson.com/papers/for.pdf

2. Theorem 6: Gaussian rounding preserves forrelation with E[Phi] = 2/pi

3. Theorem 5: k-fold Forrelation is BQP-complete
