# CHANGES.md — AI Edit Tracking Document

This file records every modification made to `run_pipeline.py` by AI-assisted cleanup.
**No features, flags, or runtime behavior were changed.** All edits are structural/cosmetic.

---

## Summary of changes

| # | File | Nature | Lines affected (approx.) |
|---|------|--------|--------------------------|
| 14 | `run_pipeline.py` + `core/speciesnet_model.py` | Move SpeciesNet utility functions to their module | ~70 lines moved |
| 1 | `run_pipeline.py` | Remove unused imports | top-level block |
| 2 | `run_pipeline.py` | Add missing `import gc` to top level | top-level block |
| 3 | `run_pipeline.py` | Remove misplaced imports inside `pretrain()` | inside function body |
| 4 | `run_pipeline.py` | Delete entire commented-out `run_wilding()` block | ~180 lines |
| 5 | `run_pipeline.py` | Remove `import gc` duplicated inside `run()` loop body | 1 line |
| 6 | `run_pipeline.py` | Remove `import re` duplicated inside `run_eval_only()` | 1 line |
| 7 | `run_pipeline.py` | Fix duplicate `'num_samples'` dict key in `final_eval_results` | 1 line |
| 8 | `run_pipeline.py` | Delete commented WildIng `argparse` block from `parse_args()` | ~30 lines |
| 9 | `run_pipeline.py` | Delete commented `run_wilding` call from `__main__` | 2 lines |
| 10 | `run_pipeline.py` | Delete `# import pdb; pdb.set_trace()` debug line from `run()` | 1 line |
| 11 | `run_pipeline.py` | Delete two commented-out comparison blocks in `run_eval_only()` | ~30 lines |
| 12 | `run_pipeline.py` | Delete commented `--model` and `--loss` argparse stubs | 4 lines |
| 13 | `run_pipeline.py` | Delete commented duplicate `--lora_interpolate` argparse stub | 2 lines |

---

## Detailed change log

### Change 1 — Import block cleanup (top-level)

**Before:** The import block contained three unused imports:
```python
import code
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
```
`import gc` was absent from the top level (it was buried inside `run()` as an inline import).

**After:** Removed the three unused imports. Added `gc` to the top-level block. Reordered to
follow stdlib → third-party → local convention (alphabetical within each group).

**Why safe:** `code`, `patches`, and `LinearSegmentedColormap` have no call sites in the file.
`matplotlib.pyplot` (which _is_ used for plotting) is kept.

---

### Change 2 — Misplaced imports removed from `pretrain()`

**Before:** Inside the body of `pretrain()` (Oracle training mode), two module-level imports
appeared as inline statements:
```python
from collections import defaultdict
import random
```
Both are already imported at the top of the file.

**After:** The two inline lines were deleted. `defaultdict` and `random` remain available from
the top-level import; no runtime behavior changes.

---

### Change 3 — Commented-out `run_wilding()` deleted

**Before:** Approximately 180 lines of commented Python code implementing WildIng training
(Santamaria et al. 2026) lived between `pretrain()` and `run()`. The function was never called
in any active code path — its only call site was also commented out in `__main__`.

**After:** The entire block (from `# def run_wilding(args):` to the line before `def run(args):`)
was deleted. WildIng source code remains in `core/wilding.py`, `core/wilding_dataset.py`,
and `precompute_vlm_descriptions.py`; it can be restored from git history if needed.

---

### Change 4 — Inline `import gc` removed from `run()` loop body

**Before:** Inside the `for i in range(start_epoch, len(ckp_list)):` loop in `run()`, after
continual-learning processing, there was:
```python
import gc
gc.collect()
```

**After:** The `import gc` line was removed. `gc.collect()` still executes; `gc` is now
imported once at the top of the file (Change 1). Identical runtime behavior.

---

### Change 5 — Inline `import re` removed from `run_eval_only()`

**Before:** At the start of the wandb initialisation block in `run_eval_only()`:
```python
import re
match = re.search(r"pipeline/([^/]+)/([^/]+)/([^/]+)", args.save_dir)
```

**After:** `import re` line deleted. `re` is imported at the module top level. `re.search`
call is unchanged.

---

### Change 6 — Duplicate `'num_samples'` key fixed in `run()`

**Before:**
```python
final_eval_results[ckp] = {
    'num_samples': len(preds_arr),
    'accuracy': float(acc),
    'balanced_accuracy': float(balanced_acc),
    'loss': float(eval_loss),
    'num_samples': len(preds_arr)   # ← duplicate key (Python silently drops first)
}
```

**After:** The second (redundant) `'num_samples'` entry was removed. The dict now has four
unique keys. The final value that Python evaluated was identical in both cases, so results
are numerically unchanged.

---

### Change 7 — Commented WildIng argparse block deleted from `parse_args()`

**Before:** A 30-line block of commented `parser.add_argument(...)` calls for WildIng
flags (`--wilding`, `--wilding_class_desc`, `--wilding_alpha`, etc.) existed inside
`parse_args()`.

**After:** The entire commented block was deleted. WildIng arguments are not registered and
will not appear in `--help` output (they were already invisible to the argument parser
because they were commented out).

---

### Change 8 — Commented `run_wilding` call deleted from `__main__`

**Before:**
```python
if args.eval_only:
    run_eval_only(args)
# elif getattr(args, 'wilding', False):
#     run_wilding(args)
else:
    run(args)
```

**After:** The two commented lines were removed. The dispatch logic is now simply:
```python
if args.eval_only:
    run_eval_only(args)
else:
    run(args)
```

---

### Change 9 — `# import pdb; pdb.set_trace()` removed from `run()`

A single commented-out debug breakpoint line inside the CL module processing block was
removed. It was `# import pdb; pdb.set_trace()` with no surrounding context that depended on it.

---

### Changes 10–11 — Two identical commented comparison blocks removed from `run_eval_only()`

Two ~15-line commented-out blocks that computed `improvement_over_` metrics across
checkpoints were removed. They appeared in both the `accu_eval` branch and the standard
eval branch. The blocks were never enabled (always commented) and the variables they
referenced (`eval_results[pre_ckp][ckp]`) would have raised `KeyError` in the `accu_eval`
branch in any case.

---

### Change 14 — SpeciesNet utilities moved to `core/speciesnet_model.py`

**Before:** `get_speciesnet_skip_classes()` and `filter_dataset_by_class_names()` were defined
as module-level functions inside `run_pipeline.py` (~70 lines), despite being purely
SpeciesNet-specific logic.

**After:** Both functions were appended to `core/speciesnet_model.py` (the natural home for
all SpeciesNet code). `run_pipeline.py` now has a single import line instead:
```python
from core.speciesnet_model import get_speciesnet_skip_classes, filter_dataset_by_class_names
```

All call sites (`get_speciesnet_skip_classes` and `filter_dataset_by_class_names`) in
`run()` are unchanged. `core/model.py` already imported `SpeciesNetClassifier` lazily from
the same module — this change makes the coupling fully explicit.

---

### Changes 12–13 — Orphan commented argparse stubs deleted

Removed:
- Commented `--model` argparse stub (superseded by `--pretrained_weights`)  
- Commented `--loss` argparse stub (superseded by `--loss_type`)  
- Commented duplicate `--lora_interpolate` float stub (the active `store_true` version is kept)

---

## Current logic of `run_pipeline.py`

### Top-level structure

```
imports (stdlib → third-party → local)

worker_init_fn()          deterministic DataLoader worker seeding
setup_logging()           builds save_dir from PETL method + loss + text encoder names
get_speciesnet_skip_classes()   returns class names marked ["skip"] in speciesnet yaml
filter_dataset_by_class_names() removes skip-class samples from a CkpDataset
pretrain()                Oracle mode: train once on all data (ckp_-1 + ckp_1)
run()                     main CL streaming loop over chronological checkpoints
run_eval_only()           loads saved .pth checkpoints and evaluates per checkpoint
parse_args()              all CLI arguments (no WildIng flags)
__main__                  seed → logging → run_eval_only() or run()
```

### `run()` high-level flow

1. Init wandb, GPU monitor
2. Load class names from train + test JSONs (+ optional rare path)
3. Build `CkpDataset` for train and eval
4. Optionally pretrain (Oracle mode) — calls `pretrain()`
5. Init OOD / AL / CL modules
6. **For each chronological checkpoint:**
   a. Subset training data up to previous checkpoint
   b. OOD detection → `ood_mask`
   c. Active learning → `al_mask` (may equal `ood_mask`)
   d. Build AL summary dict
   e. Run `cl_module.process(...)` — trains model for this checkpoint
   f. Evaluate on current checkpoint test data
   g. Log metrics → wandb; save predictions and masks
7. Save `final_training_summary.json` and `final_image_level_predictions.json`

### `run_eval_only()` high-level flow

1. Validate `--model_dir` exists
2. Optionally load `final_training_summary.json` for weight-sanity checking
3. Load class names (same logic as `run()`)
4. Build `CkpDataset` (eval only)
5. **Training-mode detection** (checked against `model_dir`):
   - `pretrain_best_model.pth` present → `upper_bound` mode (one model, all checkpoints)
   - `ckp_X_best_model.pth` files present → `accumulative` mode (per-checkpoint models)
6. **For each checkpoint** in `model_file_mapping`:
   - Load state dict; handle head shape mismatches for expanded-head runs
   - Optionally apply model/head/LoRA interpolation
   - Evaluate; log metrics; optionally save predictions
7. Save `eval_only_summary.json`; optionally calibration AUC curves

### Key CLI flags reference

| Flag | Effect |
|------|--------|
| `--pretrained_weights bioclip2` | Use BioCLIP-2 backbone (default) |
| `--lora_bottleneck 8` | Enable LoRA with rank 8 (adaptation recipe) |
| `--loss_type bsm` | Balanced Softmax loss (adaptation recipe) |
| `--full` | Full fine-tuning (FFT) of visual encoder |
| `--text head` | Only train classification head (default) |
| `--text full` | Full text encoder fine-tuning |
| `--text lora` | LoRA on text encoder |
| `--eval_only` | Skip training; evaluate saved checkpoints |
| `--resume` | Resume training from last saved checkpoint |
| `--wandb` | Enable Weights & Biases logging |
| `--eval_per_epoch` | Validate on test set after each epoch |
| `--save_best_model` | Checkpoint the epoch with best balanced accuracy |
| `--accu_eval` | After each training step, evaluate on all future checkpoints too |

CL method is set in the YAML config under `cl_config.method` — not as a CLI flag.
