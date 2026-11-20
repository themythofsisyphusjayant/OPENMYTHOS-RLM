# OpenMythos

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/themythofsisyphusjayant/OPENMYTHOS-RLM)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A recurrent language model, built from scratch, designed to train on a
single NVIDIA T4 (16 GB) in Google Colab.

The research question: can a model that reuses one block across iterations
buy effective depth more cheaply than a wider model with the same parameter
count? This repo is the test of that question — a small, fully verified
implementation, two pilot runs, and a 1B roadmap.

## Architecture

Prelude → Recurrent Block × N → Coda, with the block's weights shared across
iterations:

- **Prelude**: byte-level token embeddings (vocab 260, no tokenizer training
  needed).
- **Recurrent block** (shared): grouped-query attention + SwiGLU FFN (or
  sparse MoE), pre-norm residuals, RoPE position embeddings.
- **Coda**: final RMSNorm + LM head (optionally tied).

Two engineering consequences of the design are handled explicitly:

1. **Truncated BPTT** — gradients flow through at most
   `grad_recurrence_steps` iterations (state detached between chunks).
2. **Per-iteration KV caches** — each iteration attends over a different
   input distribution, so generation keeps one cache per iteration depth
   (see `openmythos/recurrent.py`).

## Results

Every module is shape-checked and the whole pipeline runs end to end. The
experimental record is in `research/`, and it is reported honestly — the
headline hypothesis did not survive the first pilot.

**44M run (Stage 0, TinyStories).** The 44M model was trained against a
matched-parameter baseline. On TinyStories loss, test-time depth
extrapolation did not help: evaluating at depth 8 was significantly worse
than the trained depth 4 (paired difference +0.106 loss, 95% CI
[+0.101, +0.110]). Full numbers, confidence intervals, compute, and
limitations are in
[`research/44m_experiment_report.md`](research/44m_experiment_report.md)
§4.

**MiniMythos 125M (reported, not reproduced).** A second data point, from a
report rather than a run in this repo: zero-shot HellaSwag 29.5% → 30.0%
(8 → 16 loops), PIQA flat at 50.0% — which is chance level on a 2-choice
task, so it shows no PIQA knowledge rather than robustness. See
[`research/minimythos_125m_addendum.md`](research/minimythos_125m_addendum.md).

Taken together, depth extrapolation currently looks scale- and
task-dependent, not settled either way. The 1B loop-curriculum run is the
experiment that would decide it, and it has not been run.

## Quick start

Requires Python 3.10+ and `torch>=2.2`, `numpy>=1.24`, `tqdm>=4.64`
(`requirements.txt`).

```bash
pip install -r requirements.txt

# 1. Smoke test: does the model run? (~1M params, CPU, seconds)
python -m openmythos.benchmark --preset tiny_debug --mode train --steps 5

# 2. Hardware check: parameter count + VRAM estimate for the real model
python -m openmythos.benchmark --preset mini_60m --mode params
python -m openmythos.benchmark --preset mini_60m --mode vram

# 3. Train (on a T4; see COLAB.md)
python -m openmythos.train --preset mini_60m \
    --train-data data/train.txt --val-data data/val.txt

# 4. Evaluate / generate
python -m openmythos.evaluate --ckpt checkpoints/final.pt --text data/val.txt
```

`data/` and `checkpoints/` are not in the repo; bring your own training text
and the run creates its checkpoints.

## Presets

| Preset | Params | Purpose |
|---|---|---|
| `tiny_debug` | ~1M | CPU smoke tests: forward/backward/generate in seconds |
| `mini_60m` | ~62M | Reference model for a single T4 |

Run `python -m openmythos.benchmark --preset mini_60m --mode params` for the
exact count.

## Layout

| File | Purpose |
|---|---|
| `openmythos/config.py` | Model/train configs, `tiny_debug` and `mini_60m` presets |
| `openmythos/tokenizer.py` | Byte-level tokenizer (no training step) |
| `openmythos/embeddings.py` | RMSNorm, token embeddings |
| `openmythos/attention.py` | GQA + RoPE + SDPA (FlashAttention when available), KV cache |
| `openmythos/moe.py` | SwiGLU FFN, sparse MoE with load-balancing loss |
| `openmythos/recurrent.py` | Shared recurrent block, truncated BPTT, per-iteration caches |
| `openmythos/model.py` | Prelude → RecurrentStack → Coda |
| `openmythos/dataset.py` | Text chunking + dataloaders |
| `openmythos/optimizer.py` | AdamW, selective weight decay |
| `openmythos/scheduler.py` | Linear warmup + cosine decay |
| `openmythos/trainer.py` | Loop: grad accumulation, AMP, clipping, ckpt save/resume |
| `openmythos/inference.py` | Generation with per-iteration KV caches |
| `openmythos/evaluate.py` | Perplexity CLI |
| `openmythos/benchmark.py` | Params / VRAM / tok-s smoke tests |
| `openmythos/export.py` | HF-style export (config.json + weights) |
| `openmythos/train.py` | Training CLI entry point |
| `tests/test_all.py` | Test suite |
| `COLAB.md`, `colab_t4.ipynb` | Colab setup for a T4 |
| `configs/` | 1B and 3B configs (see below) |
| `research/` | Experiment reports, battle plan, competitive survey, beginner explainers |
| `patches/` | QK-Norm and sliding-window interleave, documented against upstream |
| `experiments/pilot_cpu.py` | CPU pilot |

## Reproducibility

- All randomness is seeded (`TrainConfig.seed`); dataloader shuffling uses a
  seeded generator.
- Checkpoints store model, optimizer, scaler, and RNG states — resume with
  `--resume checkpoints/latest.pt`.
- Every run appends JSONL metrics to `checkpoints/loss.jsonl`.

## Roadmap: 1B

The T4-scale pilots validate the recurrent-depth idea as an implementation.
The milestone is a 1B-parameter OpenMythos trained and evaluated against the
open 1B class: TinyLlama-1.1B, SmolLM2-1.7B, OLMo-2-1B, Llama-3.2-1B,
Qwen2.5-1.5B.

- `configs/mythos_1b.py` — 1B config, verified at 1,056,094,338 params
  (forward pass smoke-tested on CPU): GQA 16/4, 64 experts / 8 active,
  64K custom BPE, 4K context, 16 max loop iterations, loop-depth
  curriculum. Config only — the training run has not happened.
- `configs/mythos_3b.py` — a 3B config, same family.
- `research/1b_battle_plan.md` — the plan: targets, data mix, recipe,
  compute stages.
- `research/3b_landscape.md` — competitive survey (12 models) behind the
  plan.
- `research/44m_experiment_report.md` — Stage-0 evidence (see Results).
- `research/minimythos_125m_addendum.md` — the reported 125M second data
  point.

Two caveats, stated plainly: the 1B config imports a model module
(`open_mythos.main`) that is not in this repo, and the loop-curriculum
training code it refers to is not here either — the config is verified for
its parameter count, not as a runnable training setup. And the
extrapolation claim the roadmap rests on — train with n_loops ≤ 8, sweep
4→16 at eval — is unproven: the 44M run contradicts it on TinyStories loss
and the 125M report supports it only weakly on HellaSwag.

Beginner explainers for the 44M run, MiniMythos 125M, and the 1B roadmap
are in `research/` (`*-explained-for-beginners.md`).

## License

MIT — see [LICENSE](LICENSE).
