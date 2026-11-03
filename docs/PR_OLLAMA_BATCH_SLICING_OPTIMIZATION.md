# PR: Fix Ollama Embedding Batch-Slicing Defect & Make Batch Size Configurable

**Branch:** `fix/ollama-embedding-batch-slicing`  
**Target Base:** `origin/main` (`https://github.com/sachinks/tutor`)  
**Author:** Sai Dheeraj  
**Date:** November 3, 2026  
**Status:** **Ready for Review**

---

## 1. Problem Statement & Audit Finding

In [`ai/tutor_ai/providers/ollama.py`](file:///c:/Users/Sai%20Dheeraj/Downloads/tutor-main/tutor-latest/ai/tutor_ai/providers/ollama.py#L86-L87), the `embed()` method partitioned incoming texts into batches using `self.max_batch_size` as the range step, but used the hardcoded constant `MAX_EMBED_BATCH` (32) to compute the slice end:

```python
# BEFORE (Defective):
for start in range(0, len(texts), self.max_batch_size):
    batch = [prefix + t for t in texts[start : start + MAX_EMBED_BATCH]]  # BUG: hardcoded to 32
```

### Impact of the Defect:
Whenever `max_batch_size` was configured to any value other than 32 (e.g. `max_batch_size=16` on lower-memory GPUs or CPUs):
- **Iteration 1 (`start=0`):** slices `texts[0:32]` (32 items)
- **Iteration 2 (`start=16`):** slices `texts[16:48]` (32 items)
- **Result:** Elements 16–31 were processed and embedded twice! For a 48-item list, 80 embeddings were generated instead of 48, corrupting vector arrays and throwing `ProviderError("Ollama returned a different number of embeddings")` when trailing slices misaligned.

---

## 2. Proposed Changes

1. **Correct Slicing Boundary in `OllamaProvider.embed()` (`ai/tutor_ai/providers/ollama.py`):**
   ```python
   # AFTER (Fixed):
   for start in range(0, len(texts), self.max_batch_size):
       batch = [prefix + t for t in texts[start : start + self.max_batch_size]]
   ```
   Now `self.max_batch_size` is used consistently for both iteration step and slice length.

2. **Configurable Batch Size in `Settings` & `factory.py` (`ai/tutor_ai/settings.py`, `factory.py`):**
   - Added `ollama_max_batch_size: int = Field(32, alias="TUTOR_OLLAMA_MAX_BATCH_SIZE", ge=1, le=64)` to `Settings`.
   - Wired `max_batch_size=settings.ollama_max_batch_size` into `build_provider()` in `factory.py`.

3. **Environment Documentation (`.env.example`):**
   - Documented `TUTOR_OLLAMA_MAX_BATCH_SIZE=32` with guidance for resource-constrained environments.

4. **Dedicated Test Suite (`ai/tests/unit/test_batch_slicing_optimization.py`):**
   - 9 test cases covering non-overlapping slicing on `max_batch_size=16`, arbitrary batch sizes (7, 1, 10, 32, 64), clamping invariants, and Settings validation.

5. **Local Ollama Benchmark (`benchmark_batch_slicing.py`):**
   - Benchmarks 32 passages across batch sizes 8, 16, and 32 against local Ollama (`nomic-embed-text`), verifying zero vector duplication.

---

## 3. Verification & Test Results

### 3.1 Dedicated Test Suite
Running `pytest ai/tests/unit/test_batch_slicing_optimization.py`:
```text
============================= test session starts =============================
platform win32 -- Python 3.14.0, pytest-8.4.2, pluggy-1.6.0
collected 9 items

ai/tests/unit/test_batch_slicing_optimization.py::test_batch_slicing_no_overlap_on_batch_size_16 PASSED [ 11%]
ai/tests/unit/test_batch_slicing_optimization.py::test_batch_slicing_across_various_batch_sizes[7-20-expected_batch_lengths0] PASSED [ 22%]
ai/tests/unit/test_batch_slicing_optimization.py::test_batch_slicing_across_various_batch_sizes[1-5-expected_batch_lengths1] PASSED [ 33%]
ai/tests/unit/test_batch_slicing_optimization.py::test_batch_slicing_across_various_batch_sizes[10-25-expected_batch_lengths2] PASSED [ 44%]
ai/tests/unit/test_batch_slicing_optimization.py::test_batch_slicing_across_various_batch_sizes[32-35-expected_batch_lengths3] PASSED [ 55%]
ai/tests/unit/test_batch_slicing_optimization.py::test_batch_slicing_across_various_batch_sizes[64-100-expected_batch_lengths4] PASSED [ 66%]
ai/tests/unit/test_batch_slicing_optimization.py::test_batch_size_clamping PASSED [ 77%]
ai/tests/unit/test_batch_slicing_optimization.py::test_settings_max_batch_size_configuration PASSED [ 88%]
ai/tests/unit/test_batch_slicing_optimization.py::test_factory_passes_max_batch_size PASSED [100%]

============================= 9 passed in 10.57s ==============================
```

### 3.2 Full Unit Suite Regression Check
All 163 unit tests pass with zero regressions:
```text
============================= 163 passed in 10.82s =============================
```

### 3.3 Benchmark Execution (Local Ollama: `nomic-embed-text`)
```text
================================================================================
  OLLAMA EMBEDDING BATCH-SLICING OPTIMIZATION BENCHMARK
================================================================================

Evaluating 32 text passages across configured batch sizes against local Ollama:

  [Batch Size  8] -> 4 batches | Output Vectors: 32 | Time: 10331.99 ms (322.87 ms/item)
  [Batch Size 16] -> 2 batches | Output Vectors: 32 | Time: 8238.02 ms (257.44 ms/item)
  [Batch Size 32] -> 1 batches | Output Vectors: 32 | Time: 6369.46 ms (199.05 ms/item)

--------------------------------------------------------------------------------
VERIFICATION OF BUG RESOLUTION:
  1. Array Integrity: Output vector count (32) matches input text count (32) across all batch sizes.
  2. Zero Overlap: Slice step and slice boundary are strictly aligned to self.max_batch_size.
  3. Memory Scalability: Users with smaller GPUs can configure TUTOR_OLLAMA_MAX_BATCH_SIZE=16
     without vector corruption or duplicate embedding computation.
================================================================================
```

---

## 4. Instructions for Reviewer

1. Run the new test suite:
   ```bash
   pytest ai/tests/unit/test_batch_slicing_optimization.py -v
   ```
2. Verify full suite:
   ```bash
   pytest ai/tests/unit
   ```
3. Run local Ollama benchmark:
   ```bash
   python benchmark_batch_slicing.py
   ```
