# BrainSulcal

**Champollion-conditioned BrainOmni for music-evoked EEG decoding**

Preliminary work toward a general brain stress foundation model.
NeuroSpin / CEA Saclay.

---

## Overview

BrainSulcal integrates two open-source brain foundation models to test a single
scientific hypothesis: *individual differences in sulcal morphology — particularly
Heschl's gyrus depth — modulate EEG entrainment to music, and an anatomy-conditioned
model should capture this better than a generic EEG encoder.*

| Component | Source | Role |
|-----------|--------|------|
| **Champollion V1** | NeuroSpin/CEA, HuggingFace | 56 sulcal regional embeddings from T1 MRI |
| **BrainOmni** | OpenTSLab, NeurIPS 2025 | EEG/MEG foundation model (BrainTokenizer + SpatioTemporal backbone) |
| **BrainSulcal** | This repo | Learns to inject the sulcal prior into BrainOmni |

Both pre-trained encoders are **always frozen**. Only a small set of bridging
components (~20–50 M parameters) is trained.

---

## Scientific hypothesis

```
Champollion sulcal depth (56 regions)
        │
        ▼
  SulcalAggregator ──► z_sulcal ∈ R²⁵⁶  (global subject fingerprint)
        │                    │
        │           ┌────────┴─────────┐
        │           ▼                  ▼
        │  channel_bias            prefix_proj
        │  (B,1,1,n_dim)           (B,1,lm_dim)
        │       │                       │
        ▼       ▼                       │
     BrainOmni.encode(EEG)              │
     neuro_emb += channel_bias          │
        │                               │
        ▼                               ▼
   z_eeg ∈ R⁵¹²  ◄──── mean-pool ◄─── prepend virtual channel
        │
        ▼
   [z_sulcal ‖ z_eeg] → fusion_mlp → task heads
                                       │
                                       ├─► logits_valence
                                       └─► logits_arousal
```

The **channel embedding bias** is the key surgical intervention: a subject-specific
offset `(B, 1, 1, n_dim)` is added to BrainOmni's per-channel positional (`neuro`)
embeddings before the projection layer.  Setting this bias to zero yields output
numerically identical to vanilla BrainOmni — verified by a unit test.

---

## Repository structure

```
brainsulcal/
├── priors/
│   ├── champollion_wrapper.py   # Loads pre-computed Champollion CSV embeddings
│   ├── sulcal_aggregator.py     # CLS-token transformer over 56 regions → z_sulcal
│   └── region_index.py          # Region-name → index mapping
├── dynamics/
│   ├── brainomni_wrapper.py     # Frozen BrainOmni + channel_bias injection point
│   └── moe_router_bias.py       # MLP: z_sulcal → (B,1,1,n_dim) channel bias
├── fusion/
│   └── prefix_fusion.py         # Prefix token utilities
├── data/
│   ├── daly_music.py            # Daly et al. EEG+fMRI music dataset (OpenNeuro ds002725)
│   ├── musin_g.py               # MUSIN-G dataset (OpenNeuro ds003774)
│   ├── montage_utils.py         # pos / sensor_type tensors for BrainOmni API
│   └── preprocessing.py         # MNE pipeline: filter, downsample, epoch, reject
├── training/
│   ├── trainer.py               # Training loop — differential LRs, gradient clipping, wandb
│   ├── losses.py                # CrossEntropy + InfoNCE alignment loss
│   └── metrics.py               # Accuracy, F1-macro, AUROC
├── evaluation/
│   ├── linear_probe.py          # Frozen backbone → linear classifier
│   ├── cross_subject.py         # Leave-one-subject-out (LOSO) evaluation
│   └── ablation.py              # 4-condition ablation suite
└── model.py                     # BrainSulcal top-level model

scripts/
├── 00_download_data.py          # OpenNeuro download (Daly + MUSIN-G)
├── 01_precompute_champollion.py # Champollion embeddings → data/processed/champollion/
├── 02_precompute_brainomni.py   # BrainOmni tokenizer cache → data/processed/brainomni/
├── 03_train.py                  # Main training entry point
├── 04_evaluate.py               # Full LOSO + ablation evaluation
└── 05_visualize.py              # UMAP, attention heatmaps, ROC curves

configs/
├── default.yaml
├── daly_music.yaml
└── ablation.yaml

external/
├── BrainOmni/                   # git submodule (OpenTSLab/BrainOmni)
└── champollion_pipeline/        # git submodule (neurospin/champollion_pipeline)
```

---

## Trainable vs frozen components

| Component | Parameters | Frozen |
|-----------|-----------|--------|
| Champollion V1 (56 regional encoders) | — | Yes |
| BrainOmni (BrainTokenizer + backbone) | ~300 M | Yes |
| **SulcalAggregator** (2-layer pre-norm transformer) | ~2 M | No |
| **MoERouterBias** (MLP → channel_bias) | ~0.5 M | No |
| **Prefix projection** (Linear) | ~0.1 M | No |
| **Fusion MLP + task heads** | ~0.5 M | No |

---

## Data

### Primary — Daly music dataset (OpenNeuro `ds002725`)
- 21 subjects with paired EEG + fMRI, 114 EEG-only
- 31-channel EEG, 5000 Hz (MRI-compatible), continuous valence/arousal via FEELTRACE
- Preprocessing: AAS gradient artefact removal → 1–40 Hz bandpass → 250 Hz downsample → epoch → average-reference

### Supplementary — MUSIN-G (OpenNeuro `ds003774`)
- 20 subjects, 128 channels, 250 Hz
- Familiarity/enjoyment ratings, 12 genres — used for sulcal aggregator pre-training

### Champollion embeddings
Pre-computed and cached before training:
```
data/processed/champollion/{subject_id}.npy        # (56, embedding_dim)
data/processed/champollion/{subject_id}_mask.npy   # (56,) bool
```
Subjects without T1 MRI get zero embeddings + all-False mask (ablation condition).

---

## Setup

### Prerequisites
- [Pixi](https://pixi.sh) — manages all dependencies. Do **not** use bare `pip install`.

```bash
# 1. Clone with submodules
git clone --recurse-submodules https://github.com/neurospin/brainsulcal.git
cd brainsulcal

# 2. Install all dependencies
pixi install

# 3. Install BrainOmni as an editable package
pixi run install-brainomni

# 4. Verify
pixi run --environment dev test-fast
```

> **Champollion pipeline** uses its own Pixi environment with BrainVISA/Morphologist
> dependencies that cannot be merged into this one. Always run it via its own environment:
> `cd external/champollion_pipeline && pixi run install-all`

---

## Reproducing results

```bash
# Download datasets
pixi run download-daly
pixi run download-musing   # optional

# Pre-compute embeddings (run once — results are cached)
pixi run precompute-champollion   # requires T1 MRIs + champollion_pipeline
pixi run precompute-brainomni     # tokenizer is frozen & deterministic

# Train
pixi run train

# Evaluate (LOSO + ablation)
pixi run evaluate

# Visualize (UMAP, attention heatmaps, ROC curves)
pixi run visualize
```

---

## Evaluation protocol

**Leave-one-subject-out (LOSO)** — the only statistically valid cross-subject
protocol at N = 21. Train on 20 subjects, evaluate on the held-out subject,
repeat 21 times. Report mean ± std.

> All normalization statistics (z-score, min-max) are fit on training subjects only.
> Test subject data never touches the normalization pipeline.

### Ablation conditions

| Condition | Sulcal prior | Channel bias | Prefix token |
|-----------|:-----------:|:------------:|:------------:|
| BrainOmni only (baseline) | — | — | — |
| + Prefix only | Yes | — | Yes |
| + Channel bias only | Yes | Yes | — |
| Full BrainSulcal | Yes | Yes | Yes |

### Baselines to beat

| Model | Valence accuracy |
|-------|-----------------|
| Chance | 50% |
| SVM on spectral features | ~60–65% |
| Bidirectional LSTM (EEG + fMRI) | 71.8% |
| **BrainSulcal target** | **> 70%** |

---

## Training notes

- BrainOmni tokenizer outputs are **cached to disk** before the training loop
  (tokenizer is frozen and deterministic — recomputing it every epoch wastes ~60 %
  of forward-pass time).
- Differential learning rates:
  - `SulcalAggregator`: 1e-4
  - `MoERouterBias` (channel bias): 5e-5 — slower to preserve BrainOmni routing
  - `classification_head`: 1e-3
- `channel_bias` final layer is initialised with `std = 1e-3` so training starts
  with near-zero bias (preserves BrainOmni pre-trained behaviour at initialisation).
- Monitor `router_bias_entropy` in wandb. If entropy drops below 0.5 bits in the
  first 100 steps, reduce `lr_moe_router_bias` by 10×.

---

## Key invariants (enforced by tests)

```python
# 1. Pre-trained encoders are always frozen
assert not any(p.requires_grad for p in brainomni_wrapper.parameters())

# 2. Zero channel_bias → numerically identical to vanilla BrainOmni
channel_bias_net.zero_()
assert torch.allclose(conditioned_output, vanilla_output, atol=1e-5)

# 3. Output shapes
assert z_sulcal.shape == (B, 256)
assert z_eeg.shape    == (B, 512)

# 4. Missing subjects handled gracefully (no NaNs with all-zero / all-False input)
assert not torch.any(torch.isnan(logits))
```

Run the full test suite:
```bash
pixi run --environment dev test          # all tests
pixi run --environment dev test-fast     # fail-fast
pixi run --environment dev test-cov      # with HTML coverage report
```

---

## Expected outputs

```
runs/{experiment}/
├── checkpoints/best.pt
├── results/
│   ├── loso_results.csv         # per-subject valence / arousal accuracy
│   └── ablation_table.csv       # 4-condition ablation
└── figures/
    ├── ablation_bar.png
    ├── router_bias_umap.png      # subjects clustered by routing pattern
    ├── attention_heatmap.png     # sulcal region attention weights
    └── roc_curves.png
```

---

## Experiment tracking

Weights & Biases. Key metrics logged:

- `train/loss_cls`, `train/loss_align`, `train/loss_total`
- `val/accuracy_valence`, `val/accuracy_arousal`, `val/f1_macro`, `val/auroc`
- `model/router_bias_entropy` (per BrainOmni block)
- `model/sulcal_attn_heschl` (attention weight on Heschl's gyrus regions)
- Gradient norms per trainable component group

---

## References

- **Champollion V1**: `neurospin/Champollion_V1` (HuggingFace), `neurospin/champollion_pipeline` (GitHub)
- **BrainOmni**: Xiao et al., NeurIPS 2025, arXiv 2505.18185, [OpenTSLab/BrainOmni](https://github.com/OpenTSLab/BrainOmni)
- **Daly music dataset**: OpenNeuro `ds002725`, Daly et al. *Scientific Reports* 2019, doi:10.1038/s41598-019-45105-2
- **Daly affective physiology**: *Scientific Data* 2020, doi:10.1038/s41597-020-0507-6
- **MUSIN-G**: OpenNeuro `ds003774`, Miyapuram et al. 2022
- **Music EEG + fMRI decoding**: Daly et al. *Scientific Reports* 2023, doi:10.1038/s41598-022-27361-x
- **Heschl's gyrus and musical aptitude**: Schneider et al. *Nature Neuroscience* 2005
- **GFL (EEG graph forces)**: Sarkis et al. 2025, HAL hal-05058873
