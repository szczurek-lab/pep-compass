# BattleAMP Black Box Usage Guide

This guide explains how to use the BattleAMP Black Box for peptide optimization tasks in the pep-compass framework.

## Overview

The BattleAMP Black Box provides an interface for optimizing antimicrobial peptides using the BattleAMP predictor. It follows the same interface as other black boxes in the framework (like APEX) but uses BattleAMP's specialized model for antimicrobial activity prediction.

## Quick Start

### Basic Usage

```python
import numpy as np
from pep_compass.optimization.black_box.battleamp_black_box import BattleAMPBlackBox

# Initialize the black box
black_box = BattleAMPBlackBox(device="cpu", batch_size=32)

# Prepare input sequences as character arrays
sequences = [
    list("KLLLKLLKKLLKLLK"),
    list("FLPIIAKLLGLL"),
    list("WLGHFTVRK")
]

# Pad sequences to same length
max_len = max(len(seq) for seq in sequences)
padded_sequences = []
for seq in sequences:
    padded = seq + ['A'] * (max_len - len(seq))
    padded_sequences.append(padded)

x = np.array(padded_sequences)

# Get predictions
predictions = black_box(x)
print(f"Predictions shape: {predictions.shape}")
print(f"Scores: {predictions.flatten()}")
```

### Integration with Optimization Frameworks

The BattleAMP Black Box is designed to work seamlessly with optimization frameworks like POLI:

```python
from poli import objective_factory

# Create objective function
f, x0, y0 = objective_factory.create(
    name="battleamp",
    observer=...,
    **kwargs
)

# Use with your favorite optimizer
optimizer = YourOptimizer()
optimizer.optimize(f, x0, y0)
```

## API Reference

### BattleAMPBlackBox

**Constructor Parameters:**
- `device` (str, default="cpu"): Device for computation ("cpu" or "cuda")
- `batch_size` (int, default=None): Batch size for processing sequences
- `parallelize` (bool, default=False): Enable parallel processing
- `num_workers` (int, default=None): Number of worker processes
- `evaluation_budget` (int, default=inf): Maximum number of evaluations
- `force_isolation` (bool, default=False): Force process isolation

**Key Methods:**
- `__call__(x)`: Evaluate sequences and return scores
- `get_black_box_info()`: Get metadata about the black box
- `clear_cache()`: Clear the evaluation cache

## Input Format

The black box expects input as a numpy array of character sequences:

```python
# Correct format: 2D array where each row is a sequence
x = np.array([
    ['K', 'L', 'L', 'L'],
    ['F', 'L', 'P', 'I'],
    ['W', 'L', 'G', 'H']
])
```

**Important Notes:**
- All sequences in a batch must have the same length (pad with amino acids like 'A')
- Use standard amino acid single-letter codes: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
- Maximum recommended sequence length: 25 amino acids

## Output Format

The black box returns scores as a numpy array:

```python
predictions = black_box(x)
# Shape: (n_sequences, 1)
# Values: Negative log2-transformed MIC values
# Lower (more negative) scores indicate higher antimicrobial activity
```

## Score Interpretation

- **Negative scores**: Higher antimicrobial activity (better peptides)
- **Less negative scores**: Lower antimicrobial activity
- **Scale**: Log2-transformed, so a difference of 1 unit represents a 2-fold change in MIC

## Performance Tips

1. **Batch Processing**: Use appropriate batch sizes for your hardware
   ```python
   black_box = BattleAMPBlackBox(batch_size=64)  # Adjust based on GPU memory
   ```

2. **GPU Acceleration**: Use CUDA for faster inference
   ```python
   black_box = BattleAMPBlackBox(device="cuda")
   ```

3. **Caching**: The black box automatically caches results
   ```python
   # Access cached results
   for seq, score in black_box.cache:
       print(f"{seq}: {score}")
   
   # Clear cache when needed
   black_box.clear_cache()
   ```

## Example: Optimization Workflow

```python
import numpy as np
from pep_compass.optimization.black_box.battleamp_black_box import BattleAMPBlackBox

# 1. Initialize black box
black_box = BattleAMPBlackBox(device="cpu")

# 2. Define initial sequences
initial_sequences = [
    "KLLLKLLKKLLKLLK",
    "FLPIIAKLLGLL",
    "WLGHFTVRK"
]

# 3. Convert to required format
sequences = [list(seq) for seq in initial_sequences]
max_len = max(len(seq) for seq in sequences)
padded = [seq + ['A'] * (max_len - len(seq)) for seq in sequences]
x = np.array(padded)

# 4. Evaluate
scores = black_box(x)

# 5. Find best peptide
best_idx = np.argmin(scores)
best_sequence = initial_sequences[best_idx]
best_score = scores[best_idx, 0]

print(f"Best peptide: {best_sequence}")
print(f"Best score: {best_score:.4f}")

# 6. Access evaluation history
print(f"Total evaluations: {len(black_box.cache)}")
```

## Troubleshooting

### Common Issues

1. **KeyError for unknown amino acids**
   - Solution: Use only standard amino acid codes (A-Y, excluding B, J, O, U, X, Z)

2. **Shape mismatch errors**
   - Solution: Ensure all sequences in a batch have the same length

3. **Memory errors with large batches**
   - Solution: Reduce batch_size parameter

4. **Slow performance**
   - Solution: Use GPU acceleration with `device="cuda"`

### Error Examples

```python
# ❌ Wrong: Mixed sequence lengths
x = np.array([['K', 'L'], ['F', 'L', 'P']])  # Different lengths

# ✅ Correct: Same length sequences
x = np.array([['K', 'L', 'A'], ['F', 'L', 'P']])  # Padded to same length

# ❌ Wrong: Unknown amino acid
x = np.array([['K', 'L', 'X']])  # 'X' not in BattleAMP alphabet

# ✅ Correct: Standard amino acids
x = np.array([['K', 'L', 'A']])  # All standard amino acids
```

## Comparison with APEX Black Box

| Feature | BattleAMP | APEX |
|---------|-----------|------|
| Prediction Target | Gram-negative bacteria MIC | Multi-pathogen MIC |
| Output Aggregation | Single value | Multiple pathogens (requires aggregation) |
| Model Type | CNN-based | Ensemble of models |
| Preprocessing | BattleAMP utils | APEX utils |

## References

- BattleAMP Paper: [Original BattleAMP publication]
- pep-compass Framework: [Framework documentation]
- POLI Optimization: [POLI documentation]

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify input format requirements
3. Test with the provided examples
4. Check device compatibility (CPU vs GPU)