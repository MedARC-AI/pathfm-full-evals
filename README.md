# PathFM full evaluations

`pathfm-full-evals` evaluates one pathology image encoder on the current THUNDER
leaderboard, HEST-bench, CPTAC/Patho-Bench, and PathoROB protocols. It is designed for
the MedARC shared cluster, uses at most two single-H100 jobs at once, and runs expensive
CPU probes on the CPU partition.

This is an **optimized, leaderboard-compatible protocol**, not a bitwise execution of
every upstream experiment. All task, metric, split, and numerical differences are listed
below. H-Optimus-0 reproduction results are included so that changes to this harness can
be checked against a known reference.

## Cluster contract

Benchmark inputs are read directly from canonical shared locations and are never copied
into a user's run directory.

| Suite | Shared input |
|---|---|
| THUNDER | `/data/thunder-data` |
| HEST-bench | `/data/HEST` |
| CPTAC / Patho-Bench | `/data/Patho-Bench` |
| PathoROB | `/data/pathorob` |

Pinned benchmark repositories and their `uv` environments live under
`/data/nanopath_full_evals`. Every persistent artifact produced by this repository is
isolated under `/data/$USER/pathfm-full-evals`. Node-local temporary data is isolated by
both `$USER` and `$SLURM_JOB_ID` and is removed by the producing job.

The shared benchmark installations are pinned and patched as follows:

| Upstream | Commit | Patch |
|---|---|---|
| THUNDER | `3d1cc9513fb2cfd8c4afb0d7bb9f5c4f6b69117f` | `thunder.patch` |
| HEST | `3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3` | `hest.patch` |
| Patho-Bench | `660e77044640e3d7d2f1150cc6721e97454993bf` | `patho_bench.patch` |
| PathoROB | `6583cf0b0d902c8cc032308262fa3a3befdc0687` | `pathorob.patch` |

Every submission begins with a single-H100 preflight. It verifies the selected datasets,
exact upstream commits and patches, imports, model manifest, strict checkpoint load,
preprocessing, feature shapes, finite outputs, PGD input gradients, and synthetic
regressions for the optimized THUNDER operations. Expensive stages remain held behind an
`afterok` dependency until preflight succeeds.

## Model adapter

`model.py` is the only model-specific file. The checked-in adapter evaluates a Nanopath
checkpoint by loading the architecture from the checkpoint-adjacent
`labless_source/model.py`, strict-loading its EMA weights, and applying its recorded
normalization and probe transform.

For another standard Nanopath run, change the literals at the top:

```python
CHECKPOINT = "/data/alice/nanopath/main/my-run/latest.pt"
MODEL_NAME = "my-run-fp16"
EXTRACTION_BATCH = 1024
ATTACK_BATCH = 128
```

Encoder inference uses fp16. A full H-Optimus-0 control reproduced THUNDER calibration
ECE at `3.981`, which rounds to the published `4.0`.

`MODEL_NAME` is a permanent result identity, not a display label. Preflight binds it to:

- the semantic contents of `model.py`, excluding extraction and attack batch sizes;
- the bytes of `CHECKPOINT`, its adjacent Nanopath source, and any `MODEL_ASSETS`;
- the benchmark commits, evaluation drivers, patches, and protocol version.

If any identity-bearing input changes, preflight fails and requires a new `MODEL_NAME`.
This prevents a changed checkpoint or transform from silently reusing old results.
Batch-size tuning does not require a new name.

Adapters without a local `CHECKPOINT` must declare an immutable upstream revision:

```python
MODEL_REVISION = "huggingface-commit-sha"
MODEL_ASSETS = ["/absolute/path/to/any/local/weight-file"]
```

Do not use a floating model revision such as `main`. Gated Hugging Face weights must be
downloaded before submission because compute jobs run offline.

`EvalModel` must provide:

| Member | Contract |
|---|---|
| `name` | Exactly `MODEL_NAME` |
| `classification_dim` | Width returned by `classification_features` |
| `segmentation_dim` | Per-token width returned by `segmentation_features` |
| `classification_features(images)` | Official tile representation, `[B, D]` |
| `segmentation_features(images)` | Spatial tokens only, `[B, N, D]`; exclude class/register tokens |
| `clsmean_features(images)` | PathoROB representation: global/class feature concatenated with mean patch feature |
| `transform(resize, timm_style)` | Model-specific preprocessing; `resize=False` must preserve already-224px HEST/CPTAC tiles |
| `forward(images)` | Return `classification_features(images)` |

The constructor must strict-load the intended weights, freeze the backbone, set eval
mode, and move it to CUDA. Use the model author's published readout and preprocessing;
do not assume every transformer uses its class token. Models without a spatial token
grid cannot run THUNDER segmentation or the PathoROB `clsmean` protocol without a
documented spatial representation.

The adapter smoke test can be run manually on one H100:

```bash
srun --partition=n --account=sophont --qos=high --gres=gpu:1 --cpus-per-task=4 \
  --mem=32G --time=00:10:00 \
  /data/nanopath_full_evals/repos/thunder/.venv/bin/python model.py
```

Use a separate checkout and unique `MODEL_NAME` for each concurrently evaluated model.

## Run

Run all four suites:

```bash
./submit_all.sh
./submit_all.sh --embeddings=retain
```

Run the three suites that do not require PathoROB:

```bash
./submit_all.sh --no-pathorob
./submit_all.sh --no-pathorob --embeddings=retain
```

Run one suite:

```bash
./submit_suite.sh thunder
./submit_suite.sh hest
./submit_suite.sh cptac
./submit_suite.sh pathorob
./submit_suite.sh thunder --embeddings=retain
```

Precomputed embeddings are removed by default. Pass `--embeddings=retain` to keep each
suite's persistent embeddings after its consumers finish; `--embeddings=remove` selects
the default explicitly. Job-scoped scratch data under `/tmp` is always removed.

The submitters use native Slurm arrays with `%2` for GPU work and `%8` for independent
CPU fits. Slurm schedules the next array element as a worker becomes available; there is
no repository-owned claim-file scheduler. Jobs are requeueable on cluster preemption.
An evaluation error fails loudly and blocks dependent stages. Resubmitting the same
command verifies the run manifest, skips readable completed outputs, rebuilds incomplete
caches, and resumes remaining work.

GPU jobs request one H100 and at most 16 CPUs. The full dependency graph never has more
than two GPU array elements eligible to run concurrently. Do not launch independent
suite submissions concurrently unless their combined GPU concurrency remains at most
two.

Monitor jobs and logs with:

```bash
squeue -u "$USER" -o '%.18i %.24j %.2t %.10M %R'
tail -f /data/$USER/pathfm-full-evals/logs/<job>_<array-index>_gpu.out
sacct -j <job> --format=JobID,JobName,State,Elapsed,MaxRSS,AllocTRES
```

## Work graph

| Resource | Stage | Approx. Nanopath wall time | Work |
|---|---|---:|---|
| GPU | preflight | 1 min | Manifest, data, code, environment, model, gradient, and regression validation |
| GPU | THUNDER precompute | 20–30 min | Classification embeddings for 16 datasets |
| GPU | THUNDER cached | 1 hr 45 min–2 hr 15 min | KNN, linear/calibration, and 16-shot SimpleShot |
| CPU | THUNDER cleanup | <1 min | Validate cached-probe outputs, then apply the embedding retention policy |
| GPU | THUNDER online | 4–4.5 hr | Four segmentation tasks and PGD attacks |
| CPU | THUNDER summary | 1 min | Produce the six-row leaderboard summary |
| GPU | HEST extract | 3–5 min | Extract all nine datasets in three balanced groups |
| CPU | HEST probes | 2–3 min | PCA-256 and Ridge for all nine datasets |
| CPU | HEST finalize | <1 min | Aggregate Pearson scores and apply the embedding retention policy |
| GPU | PathoROB extract | 5 min | Extract all three datasets |
| CPU | PathoROB metrics | 1 hr 45 min–2 hr | Nine dataset/metric combinations |
| CPU | PathoROB finalize | <1 min | Validate summaries and apply the embedding retention policy |
| GPU | CPTAC extract | 50–60 min | Eight balanced slide shards, capped at two concurrent H100s |
| CPU | CPTAC pool | <1 min | Pool slide means to cases |
| CPU | CPTAC probes | 3–5 min | 38 classification fits and 12 survival-alpha fits |
| CPU | CPTAC finalize | <1 min | Aggregate results and apply the embedding retention policy |

Times are measured stage wall times for ViT-S Nanopath models on the shared cluster,
excluding queue time. Array stages run tasks concurrently, and independent CPU and GPU
branches overlap, so these values should not be summed to estimate total run time.

HEST and PathoROB share one GPU array in the full run. Their CPU stages then proceed
independently while THUNDER and CPTAC use the GPUs. By default, intermediate embeddings
are removed only after every consumer has produced a readable result; the retain option
keeps them in their suite output directories.

## Results

```text
/data/$USER/pathfm-full-evals/manifests/<MODEL_NAME>.json
/data/$USER/pathfm-full-evals/thunder/outputs/res/results.csv
/data/$USER/pathfm-full-evals/hest/results/<MODEL_NAME>/aggregate.json
/data/$USER/pathfm-full-evals/cptac/<MODEL_NAME>/aggregate.json
/data/$USER/pathfm-full-evals/pathorob/results/
```

THUNDER appends `_optimized` to its model key so these results cannot be confused with
an unmodified upstream run. The manifest records exactly which model and protocol
produced every result namespace.

## Protocol differences and tradeoffs

### THUNDER

The target is the current six-column website leaderboard: KNN, linear probing,
calibration, 16-shot SimpleShot, adversarial attack, and four-dataset segmentation.
Other paper shot counts and experiments are not run.

- Classification features are extracted once and reused by KNN, linear probing, and
  SimpleShot.
- Encoder calls use fp16 and frozen features are saved as fp32.
- KNN, SimpleShot, linear-head execution, and segmentation metrics use equivalent
  vectorized PyTorch implementations. Ties and floating-point rounding can differ from
  NumPy/scikit-learn.
- Segmentation tokens use symmetric int8 quantization with one fp16 scale per token.
  This avoids hundreds of gigabytes of cache traffic but is not bitwise equivalent to
  fp32 token caching.
- Contiguous token arrays are memory-mapped from job-scoped node-local storage; small
  caches may stay on the GPU.
- Segmentation decoder matmuls use TF32. Frozen classification probes remain fp32.
- The leaderboard's point estimates are retained; 3,000-sample confidence intervals and
  online mask visualization are omitted.
- SegPath epithelial and lymphocyte tasks retain the official 9- and 21-epoch overrides;
  frozen probes and segmentation retain batch size 64.

### HEST-bench

All nine current tasks retain PCA-256, Ridge LSQR, official splits, and Pearson
correlation. GPU extraction is separated from CPU probing, the optional ResNet-50 side
baseline is omitted, and specimen embeddings follow the submission's retention policy
after aggregation.

### CPTAC / Patho-Bench

The 11 available CPTAC cohorts retain official splits, logistic-probe settings, 100 test
bootstraps, and CoxNet `alpha={0.01,0.02,0.07}`, `l1_ratio=0.5` fits. Patch embeddings are
streamed into fp64 slide sums rather than stored densely, then unweighted slide means are
pooled to cases. The combined `cptac_all/organ` task and unavailable external DHMC cohort
are not run. A known upstream CoxNet “weights too large” failure is recorded as an
explicit `null`; all other failures stop the job.

### PathoROB

Robustness index, average performance drop, and clustering are run on Camelyon, TCGA,
and Tolkach ESCA. The adapter supplies the published `clsmean` representation and
model-specific transform. Features are stored in packed per-center arrays, shared by all
three metrics, and follow the submission's retention policy only after all expected JSON
outputs parse successfully.

## Reference validation

The optimized H-Optimus-0 reproduction against the current THUNDER website is:

| THUNDER row | Website | This harness | Difference |
|---|---:|---:|---:|
| KNN macro-F1 | 81.4 | 81.4 | 0.0 |
| Linear macro-F1 | 83.8 | 83.8 | 0.0 |
| 16-shot macro-F1 | 76.2 | 76.1 | -0.1 |
| Segmentation macro-F1 | 65.2 | 65.0 | -0.2 |
| Calibration ECE | 4.0 | 4.0 | 0.0 |
| PGD macro-F1 | 43.9 | 43.9 | 0.0 |

The calibration reproduction used fp16 encoder inference and scored `3.981` before
leaderboard rounding.

HEST mean Pearson was `0.41493` versus the published `0.415`. PathoROB robustness
indices were `0.70444`, `0.81214`, and `0.91780` versus published
`0.705/0.812/0.918`. CPTAC mean classification macro-OVR AUC was `0.67299`; two of the
12 CoxNet fits produced their known explicit numerical-failure result.

Measured H-Optimus-0 wall time is approximately 12.5–13 hours for the complete graph,
excluding queue time. The checked-in ViT-S Nanopath adapter is approximately 8.2 hours.
These are two-H100 critical paths, not sums of individual task durations.
