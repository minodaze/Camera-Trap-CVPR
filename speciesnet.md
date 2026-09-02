# SpeciesNet Zero-Shot Guide

This document explains how SpeciesNet is wired into this repository, how to run the zero-shot pipeline, how the SpeciesNet alias mapping works, and what was added in `core/speciesnet_model.py` to make SpeciesNet compatible with the rest of the codebase.

The goal of this integration is simple:

1. Load a pretrained SpeciesNet checkpoint.
2. Read the SpeciesNet taxonomy labels from the provided `.labels.txt` file.
3. Map the repository's dataset classes to one or more SpeciesNet aliases.
4. Convert SpeciesNet's raw taxonomy logits back into dataset-class logits.
5. Run the existing evaluation pipeline without rewriting the rest of the project.

---

## 1. What SpeciesNet Is Doing In This Repo

SpeciesNet is used here as a **zero-shot classifier**.

That means:

- It is not trained from scratch on your camera-trap dataset.
- It is not fine-tuned by default in this wrapper.
- Instead, the pretrained SpeciesNet model predicts over its own taxonomy/classes.
- The wrapper then collapses those SpeciesNet outputs into your dataset label space using the alias map in the YAML config.

In practice, this makes SpeciesNet behave like a dataset-class classifier even though the underlying model is a taxonomy classifier.

---

## 2. Files That Matter

The SpeciesNet integration mainly touches these files:

- `Camera_Trap_Speciesnet/Camera-Trap-CVPR/core/speciesnet_model.py`
- `Camera_Trap_Speciesnet/Camera-Trap-CVPR/core/model.py`
- `Camera_Trap_Speciesnet/Camera-Trap-CVPR/core/data.py`
- `Camera_Trap_Speciesnet/Camera-Trap-CVPR/run_pipeline.py`
- `Camera_Trap_Speciesnet/Camera-Trap-CVPR/config/SpeciesNet/<dataset>/zs.yaml`
- `Camera_Trap_Speciesnet/Camera-Trap-CVPR/my_weights/speciesnet-pytorch-v4.0.1a-v1/always_crop_99710272_22x8_v12_epoch_00148.pt`
- `Camera_Trap_Speciesnet/Camera-Trap-CVPR/my_weights/speciesnet-pytorch-v4.0.1a-v1/always_crop_99710272_22x8_v12_epoch_00148.labels.txt`

The `.pt` file is the pretrained SpeciesNet checkpoint.
The `.labels.txt` file defines the SpeciesNet taxonomy class order.

The order matters a lot: the `k`-th line in the labels file must correspond to the `k`-th output logit of the checkpoint.

---

## 3. What Was Added In `core/speciesnet_model.py`

The file `core/speciesnet_model.py` is the core addition. It wraps SpeciesNet so the existing evaluation pipeline can consume it without special-case code everywhere else.

### 3.1 Main wrapper class

The main class is `SpeciesNetClassifier(nn.Module)`.

Its responsibility is to:

- load the pretrained SpeciesNet checkpoint,
- discover or reconstruct the SpeciesNet class names,
- build a mapping from dataset classes to SpeciesNet alias indices,
- forward images through SpeciesNet,
- aggregate SpeciesNet logits into dataset logits.

### 3.2 Checkpoint loading is flexible

The loader is designed to handle more than one checkpoint format:

- a full `nn.Module` saved directly with `torch.save(model, ...)`,
- a dict containing a model object,
- a dict containing a `state_dict`,
- a raw `state_dict` itself.

It also strips common prefixes like:

- `module.`
- `model.`
- `classifier.`
- `speciesnet.`
- `net.`

That makes it more tolerant of checkpoints produced under DDP or other wrappers.

### 3.3 Class name extraction

The wrapper tries to recover SpeciesNet class names in this order:

1. Try names stored in the checkpoint itself.
2. Fall back to `always_crop_99710272_22x8_v12_epoch_00148.labels.txt` in the same weight directory.

If the checkpoint is a full `GraphModule` or model object, the labels file is the normal source of taxonomy names.

### 3.4 Label file parsing is alignment-safe

The labels loader is intentionally careful about row order.

The label file contains semicolon-separated entries that look like:

```text
uuid;class;order;family;genus;species;common_name
```

The loader:

- skips only truly empty lines,
- skips comment lines,
- preserves row alignment even if a common name is missing,
- creates placeholder names when needed so logit indices stay aligned.

This is important because the output index must match the file line index.

### 3.5 Dataset-to-SpeciesNet mapping

The wrapper builds two lookup tables:

- `speciesnet_name_to_idx`: normalized SpeciesNet name -> list of logit indices
- `dataset_to_speciesnet_indices`: dataset class -> list of SpeciesNet indices

This is necessary because:

- SpeciesNet may contain duplicate common names,
- a dataset class may map to multiple aliases,
- and you may want to keep both family-level and species-level names available.

The final dataset-class logit is computed as:

- the **max** of all SpeciesNet logits for the aliases associated with that dataset class.

That design is very practical for zero-shot species labeling because one dataset class may correspond to multiple SpeciesNet names.

### 3.6 Input format handling

SpeciesNet's exported `GraphModule` expects channels-last input:

- SpeciesNet expects: `[B, H, W, C]`
- torchvision `ToTensor()` produces: `[B, C, H, W]`

The wrapper automatically permutes the tensor before inference:

```python
images = images.permute(0, 2, 3, 1).contiguous()
```

This is one of the key integration fixes in the wrapper.

### 3.7 Output normalization and debugging

SpeciesNet may return:

- a tensor,
- a tuple/list,
- or a dict with keys such as `logits`, `classification`, `classifier_logits`, or `predictions`.

The wrapper normalizes that output into a raw `[B, K]` logits tensor.

It also logs a one-time debug summary of the raw output:

- tensor shape,
- min/max/mean,
- top-10 classes for the first few samples.

This makes it much easier to confirm that:

- the checkpoint is loading correctly,
- the taxonomy alignment is correct,
- and the output distribution looks reasonable.

### 3.8 SpeciesNet is frozen

The wrapper explicitly sets:

- `eval()` mode,
- `requires_grad = False` for all parameters.

So in the current form, this is a **zero-shot / evaluation wrapper**, not a trainable SpeciesNet finetuning module.

### 3.9 Important limitation

`forward_features()` is intentionally not implemented.

That means:

- you can use this wrapper for zero-shot evaluation,
- but you should not expect it to behave like the repository's trainable backbones without adding extra work.

If you want actual finetuning later, see the suggestions in the last section.

---

## 4. How The Repo Selects SpeciesNet

The integration is activated in `core/model.py`:

- if `common_config.model == "speciesnet"`,
- the code imports `SpeciesNetClassifier`,
- reads `speciesnet_aliases` from the YAML,
- and constructs the wrapper around the pretrained checkpoint.

There is also a small compatibility fallback for the typo:

- `speciesnet_aliaes`

So if that typo appears in an older config, the code still tries to recover.

---

## 5. How The Pipeline Treats Zero-Shot SpeciesNet

The zero-shot behavior is handled in `run_pipeline.py`.

### 5.1 Skip classes

If a class is mapped to:

```yaml
speciesnet_aliases:
  some_class: ["skip"]
```

then that dataset class is excluded from SpeciesNet evaluation.

This is useful when:

- a class has no reliable SpeciesNet equivalent,
- a label is too coarse,
- or you want to remove a class from the zero-shot comparison entirely.

### 5.2 Dataset transforms for SpeciesNet

In `core/data.py`, SpeciesNet uses a different validation transform:

```python
Resize((480, 480), interpolation=InterpolationMode.BICUBIC)
ToTensor()
```

That differs from the standard 224x224 validation path.

So when you run SpeciesNet, the dataset loader is already set up to use the larger SpeciesNet-friendly input size.

---

## 6. SpeciesNet Config Layout

All SpeciesNet configs are under:

```text
Camera_Trap_Speciesnet/Camera-Trap-CVPR/config/SpeciesNet
```

Each dataset/site has its own subdirectory containing a `zs.yaml`.

Available configs currently include:

- `APN_13U/zs.yaml`
- `APN_K082/zs.yaml`
- `ENO_B06/zs.yaml`
- `ENO_C02/zs.yaml`
- `KGA_KHOLA03/zs.yaml`
- `MAD_H08/zs.yaml`
- `MTZ_E05/zs.yaml`
- `caltech_88/zs.yaml`
- `na_lebec_CA-19/zs.yaml`
- `nz_EFD_DCAMF06/zs.yaml`
- `nz_EFH_HCAMD08/zs.yaml`
- `nz_EFH_HCAME08/zs.yaml`
- `nz_EFH_HCAME09/zs.yaml`
- `nz_EFH_HCAMI01/zs.yaml`

### 6.1 Example YAML structure

The representative config looks like this:

```yaml
module_name: zs

log_path: /fs/scratch/PAS2099/Lemeng/speciesnetResult/Camera_Trap_Speciesnet/log_speciesnet/APN_13U/zs

pretrained_weights: "/fs/scratch/PAS2099/Lemeng/speciesnetResult/Camera_Trap_Speciesnet/Camera-Trap-CVPR/my_weights/speciesnet-pytorch-v4.0.1a-v1/always_crop_99710272_22x8_v12_epoch_00148.pt"

label_type: common

speciesnet_aliases:
  african bush elephant: ["african elephant"]
  kudu: ["greater kudu", "lesser kudu"]
  baboon: ["kinda baboon", "yellow baboon", "olive baboon", "baboon species", "chacma baboon"]
  warthog: ["common warthog", "desert warthog"]
  spotted hyena: ["spotted hyaena"]
  burchell's zebra: ["plains zebra", "grevy's zebra", "mountain zebra"]
  impala: ["impala"]
  common duiker: ["common duiker"]
  giraffe: ["giraffe"]

common_config:
  model: speciesnet
  train_data_config_path: config/data/APN/APN_13U/30/train.json
  eval_data_config_path: config/data/APN/APN_13U/30/test.json

  train_batch_size: 128
  eval_batch_size: 512

  optimizer_name: AdamW
  optimizer_params:
    lr: 0.000025
    weight_decay: 0.0001

  scheduler: null
  scheduler_params: null
  chop_head: false

pretrain_config:
  pretrain: false

ood_config:
  method: none

al_config:
  method: none

cl_config:
  method: none
```

### 6.2 What each key means

- `module_name: zs`
  - Marks the config as a zero-shot setup.

- `log_path`
  - Where the run output, metrics, and logs should go.

- `pretrained_weights`
  - Absolute path to the SpeciesNet checkpoint.

- `label_type`
  - Which label field to use from the dataset JSON.
  - In these configs it is usually `common`.

- `speciesnet_aliases`
  - Dataset class name -> SpeciesNet alias list.
  - This is the core mapping that translates your dataset labels into SpeciesNet names.

- `common_config.model`
  - Must be `speciesnet` to activate the wrapper.

- `train_data_config_path` / `eval_data_config_path`
  - Paths to the dataset JSON files.

- `pretrain_config.pretrain: false`
  - Indicates that this is not a normal finetuning run.

- `ood_config`, `al_config`, `cl_config`
  - Set to `none` here because the zero-shot SpeciesNet run is not using those training strategies.

---

## 7. SpeciesNet Alias File

The label file you provided is:

```text
Camera_Trap_Speciesnet/Camera-Trap-CVPR/my_weights/speciesnet-pytorch-v4.0.1a-v1/always_crop_99710272_22x8_v12_epoch_00148.labels.txt
```

This file is the source of truth for SpeciesNet class ordering.

### 7.1 Format

Each row contains a taxonomy entry. The common name is usually the last field.

Example style:

```text
uuid;class;order;family;genus;species;common name
```

Some rows may omit certain fields, and some may have missing common names. The loader preserves index alignment by inserting placeholder labels when necessary.

### 7.2 Why this file matters

The SpeciesNet wrapper builds the alias lookup by matching:

- dataset class names from your YAML,
- against normalized SpeciesNet common names from this file.

So if a class is not behaving as expected, the first things to check are:

- spelling,
- plurality,
- hyphens vs spaces,
- and whether the alias exists in the labels file exactly as expected after normalization.

### 7.3 Normalization behavior

The wrapper lowercases and trims names before matching.

So these are treated as equivalent:

- `African Elephant`
- `african elephant`
- ` african elephant `

But the alias still has to match the actual taxonomy name after normalization.

---

## 8. How To Run Zero-Shot SpeciesNet

Below is the most direct way to run a zero-shot evaluation with the current pipeline.

### 8.1 Preconditions

Make sure all of the following exist:

- the SpeciesNet checkpoint `.pt`,
- the matching `.labels.txt`,
- the dataset JSON files referenced by the config,
- the SpeciesNet config you want to run, for example `config/SpeciesNet/APN_13U/zs.yaml`.

Also make sure the `log_path` directory is writable.

### 8.2 Recommended command

Use the pipeline with the SpeciesNet config:

```bash
python run_pipeline.py \
  --c config/SpeciesNet/APN_13U/zs.yaml
```

### 8.3 If you want the pipeline to evaluate every checkpoint in a model directory

The eval-only flow can also discover checkpoint files such as:

- `pretrain_best_model.pth`
- `ckp_X_best_model.pth`

SpeciesNet zero-shot will still be used for `ckp_1`.

### 8.4 What you should see in the logs

On a successful run, the logs should show:

- checkpoint loading,
- label file parsing,
- number of SpeciesNet classes,
- number of dataset classes,
- alias-to-index mappings,
- raw logits statistics,
- top-10 debug predictions for the first batch,
- final evaluation metrics.

If an alias is missing or mismatched, the run should fail early with a helpful error message showing nearby candidate names.

---

## 9. How To Add A New SpeciesNet Dataset Config

If you want to create a new SpeciesNet zero-shot setup, the easiest path is to copy one of the existing `zs.yaml` files and edit:

1. `train_data_config_path`
2. `eval_data_config_path`
3. `log_path`
4. `speciesnet_aliases`
5. `label_type` if your dataset uses scientific names instead of common names

### 9.1 Suggested workflow

- Copy the nearest site/dataset config.
- Update the JSON paths.
- Keep `common_config.model: speciesnet`.
- Point `pretrained_weights` to the same checkpoint unless you are evaluating a different SpeciesNet release.
- Verify every dataset class has either:
  - one or more valid aliases, or
  - `["skip"]`.

### 9.2 Alias mapping advice

For best results:

- use the most common English name first,
- include known alternate spellings,
- include species-level and broader-name variants when needed,
- keep aliases lowercase for readability,
- use `skip` only when no sensible mapping exists.

---

## 10. Troubleshooting

### 10.1 Alias not found

If you see an error like:

```text
SpeciesNet alias '...' for dataset class '...' not found in SpeciesNet class names
```

then one of these is usually true:

- the alias is misspelled,
- the alias does not exist in the `.labels.txt` file,
- the class name should map to a different common name,
- the class should be marked `skip`.

### 10.2 Output head index out of bounds

If you see an index error, it usually means:

- the label file order does not match the model output order,
- or the checkpoint and label file are from different SpeciesNet releases.

This is why the `.pt` file and `.labels.txt` file should stay together in the same weight directory.

### 10.3 Wrong input shape

The wrapper automatically changes input from NCHW to NHWC.

If you modify the data pipeline, keep this assumption in mind:

- torchvision returns channel-first tensors,
- the exported SpeciesNet GraphModule expects channel-last tensors.

### 10.4 Classes that should be ignored

If you want a label excluded from zero-shot evaluation, use:

```yaml
speciesnet_aliases:
  some_label: ["skip"]
```

Then the pipeline will remove that class from the evaluation samples.

---

## 11. Suggestions For Later Finetuning SpeciesNet

The current wrapper is intentionally conservative and zero shot only. If you want to extend this into a finetuning pipeline later, here are the best next steps.

### 11.1 Decide whether you want to finetune the original SpeciesNet backbone

Right now, the wrapper does not expose `forward_features()`, and it freezes all parameters.

So for real finetuning you will likely need one of these approaches:

- unwrap the SpeciesNet backbone into a trainable module,
- load a `state_dict` into a compatible `timm` architecture,
- or build a new trainable head on top of SpeciesNet features.

If your goal is just better performance on your camera-trap classes, a lightweight adaptation head is often the fastest path.

### 11.2 Start with distillation

A strong practical strategy is:

- keep SpeciesNet frozen,
- use its logits as teacher signals,
- train a smaller student or dataset-specific head on your labeled data.

This keeps the original zero-shot behavior as a baseline while still letting the pipeline adapt to your domain.

### 11.3 Calibrate the alias aggregation

Currently the dataset class score is the max over all matching SpeciesNet alias logits.

For finetuning or calibration, you could test:

- max pooling across aliases,
- average pooling across aliases,
- log-sum-exp pooling,
- or a learned alias aggregation layer.

### 11.4 Make the label mapping explicit and versioned

As you finetune, keep a versioned mapping file for:

- dataset class name,
- SpeciesNet alias list,
- skip status,
- and any scientific/common-name conversion.

That will save a lot of time when you compare zero-shot, adapted, and finetuned runs.

### 11.5 Add domain-specific augmentation

Camera-trap data tends to benefit from:

- stronger crop/resize augmentation,
- horizontal flip where valid,
- class-balanced sampling,
- careful handling of nighttime / blur / occlusion,
- and species-specific class imbalance strategies.

### 11.6 Use a staged adaptation schedule

If you move to training, a stable schedule is:

1. Start with the SpeciesNet backbone frozen.
2. Train only a new head or adapter.
3. Unfreeze later layers only if needed.
4. Compare against the zero-shot baseline at every stage.

This makes it much easier to tell whether the finetuning is actually helping.

### 11.7 Preserve the zero-shot benchmark

Even after finetuning, keep the zero-shot SpeciesNet run as a baseline.

That gives you a clean comparison for:

- domain shift,
- alias quality,
- training data size,
- and class coverage.

---

## 12. Practical Summary

If you want the shortest possible mental model:

- `core/speciesnet_model.py` makes SpeciesNet look like the repository's existing classifier API.
- `config/SpeciesNet/<dataset>/zs.yaml` tells the pipeline which dataset classes map to which SpeciesNet aliases.
- `always_crop_99710272_22x8_v12_epoch_00148.pt` is the pretrained model.
- `always_crop_99710272_22x8_v12_epoch_00148.labels.txt` defines the taxonomy class order.
- The wrapper converts SpeciesNet logits into dataset logits by alias matching and max pooling.
- The current implementation is zero-shot and frozen, so finetuning requires a new trainable extension.

If you keep those five pieces aligned, the SpeciesNet pipeline should remain easy to run and easy to extend later.