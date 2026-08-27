#!/usr/bin/env bash
#SBATCH --output=/data/%u/pathfm-full-evals/logs/%A_%a_gpu.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --array=0-1%2
#SBATCH --time=48:00:00
#SBATCH --partition=n
#SBATCH --account=sophont
#SBATCH --qos=high
#SBATCH --requeue

set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
[[ "${EVAL_STAGE:-}" == "preflight" || "${EVAL_STAGE:-}" == "thunder_precompute" || "${EVAL_STAGE:-}" == "thunder_cached" || "${EVAL_STAGE:-}" == "hest_pathorob" || "${EVAL_STAGE:-}" == "hest_extract" || "${EVAL_STAGE:-}" == "pathorob_extract" || "${EVAL_STAGE:-}" == "cptac_extract" || "${EVAL_STAGE:-}" == "thunder_online" ]]

repo=$PWD
if [[ "$EVAL_STAGE" == "preflight" ]]; then
  [[ "$SLURM_ARRAY_TASK_COUNT" == 1 && "$SLURM_ARRAY_TASK_ID" == 0 ]]
  bash -n "$repo/submit_all.sh" "$repo/submit_suite.sh" "$repo/run_gpu.sbatch" "$repo/run_cpu.sbatch"
  [[ -x "$repo/submit_all.sh" && -x "$repo/submit_suite.sh" ]]
  /data/nanopath_full_evals/repos/thunder/.venv/bin/python "$repo/run_manifest.py" write
  preflight_suites=${PREFLIGHT_SUITES:-thunder:hest:cptac:pathorob}
  IFS=: read -r -a suites <<< "$preflight_suites"
  [[ "${#suites[@]}" -gt 0 ]]
  for suite in "${suites[@]}"; do
    case "$suite" in
      thunder)
        data=/data/thunder-data
        upstream=thunder
        commit=3d1cc9513fb2cfd8c4afb0d7bb9f5c4f6b69117f
        patch=thunder.patch
        driver=thunder_eval.py
        ;;
      hest)
        data=/data/HEST
        upstream=HEST
        commit=3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3
        patch=hest.patch
        driver=hest_eval.py
        ;;
      cptac)
        data=/data/Patho-Bench
        upstream=Patho-Bench
        commit=660e77044640e3d7d2f1150cc6721e97454993bf
        patch=patho_bench.patch
        driver=cptac_eval.py
        ;;
      pathorob)
        data=/data/pathorob
        upstream=PathoROB
        commit=6583cf0b0d902c8cc032308262fa3a3befdc0687
        patch=pathorob.patch
        driver=pathorob_eval.py
        ;;
      *)
        echo "unknown preflight suite: $suite" >&2
        exit 2
        ;;
    esac
    upstream_root=/data/nanopath_full_evals/repos/$upstream
    python=$upstream_root/.venv/bin/python
    [[ -r "$data" && -x "$data" && -x "$python" ]]
    [[ "$(git -C "$upstream_root" rev-parse HEAD)" == "$commit" ]]
    git -C "$upstream_root" apply --check --reverse "$repo/$patch"
    if [[ "$suite" == "thunder" ]]; then
      thunder_index=$(mktemp)
      trap 'rm -f "$thunder_index"' EXIT
      cp "$upstream_root/.git/index" "$thunder_index"
      GIT_INDEX_FILE="$thunder_index" git -C "$upstream_root" add -N \
        src/thunder/config/wandb/disabled.yaml
      cmp -s "$repo/$patch" <(GIT_INDEX_FILE="$thunder_index" \
        git -C "$upstream_root" diff --binary HEAD)
      rm "$thunder_index"
      trap - EXIT
      PYTHONPATH="$upstream_root/src" "$python" -c \
        "from thunder import benchmark, download_datasets, download_models, generate_splits; import h5py"
    else
      cmp -s "$repo/$patch" <(git -C "$upstream_root" diff --binary HEAD)
      if [[ "$suite" == "hest" ]]; then
        PYTHONPATH="$upstream_root/src" "$python" -c "import h5py, hest, pandas"
      elif [[ "$suite" == "cptac" ]]; then
        "$python" -c "import h5py, patho_bench, pandas, yaml"
      else
        "$python" -c "import datasets, pathorob, sklearn"
      fi
    fi
    "$python" -m py_compile "$repo/model.py" "$repo/run_manifest.py" "$repo/$driver"
    PYTHONPATH="$repo" "$python" "$repo/model.py"
  done
  /data/nanopath_full_evals/repos/thunder/.venv/bin/python "$repo/test_eval.py"
  echo "PASS suites=$preflight_suites scripts, manifests, shared data, pinned benchmark code, environments, checkpoint-native preprocessing, model interface, input gradients, and synthetic regressions"
  exit 0
fi

tasks=()
case "$EVAL_STAGE" in
  thunder_precompute)
    tasks=("thunder precompute_group 0" "thunder precompute_group 1")
    ;;
  thunder_cached)
    tasks=("thunder cached_group 0" "thunder cached_group 1")
    ;;
  hest_pathorob)
    tasks=(
      "pathorob extract tcga"
      "hest extract_group 1" "hest extract_group 0" "hest extract_group 2"
      "pathorob extract camelyon" "pathorob extract tolkach_esca"
    )
    ;;
  hest_extract)
    tasks=("hest extract_group 1" "hest extract_group 0" "hest extract_group 2")
    ;;
  pathorob_extract)
    tasks=("pathorob extract tcga" "pathorob extract camelyon" "pathorob extract tolkach_esca")
    ;;
  cptac_extract)
    for shard in {0..7}; do
      tasks+=("cptac extract $shard 8")
    done
    ;;
  thunder_online)
    tasks=(
      "thunder segmentation 1" "thunder segmentation 2"
      "thunder segmentation 3" "thunder segmentation 0"
      "thunder adversarial_group 0" "thunder adversarial_group 1"
    )
    ;;
esac

[[ "$SLURM_ARRAY_TASK_ID" -lt "${#tasks[@]}" ]]
read -r -a task <<< "${tasks[$SLURM_ARRAY_TASK_ID]}"
echo "START ${tasks[$SLURM_ARRAY_TASK_ID]}"
case "${task[0]}" in
  thunder)
    exec env PYTHONPATH=/data/nanopath_full_evals/repos/thunder/src \
      /data/nanopath_full_evals/repos/thunder/.venv/bin/python \
      "$repo/thunder_eval.py" "${task[@]:1}"
    ;;
  hest)
    exec env PYTHONPATH=/data/nanopath_full_evals/repos/HEST/src \
      /data/nanopath_full_evals/repos/HEST/.venv/bin/python \
      "$repo/hest_eval.py" "${task[@]:1}"
    ;;
  pathorob)
    exec /data/nanopath_full_evals/repos/PathoROB/.venv/bin/python \
      "$repo/pathorob_eval.py" "${task[@]:1}"
    ;;
  cptac)
    exec /data/nanopath_full_evals/repos/Patho-Bench/.venv/bin/python \
      "$repo/cptac_eval.py" "${task[@]:1}"
    ;;
esac
