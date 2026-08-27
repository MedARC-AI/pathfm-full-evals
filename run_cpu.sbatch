#!/usr/bin/env bash
#SBATCH --job-name=eval_cpu
#SBATCH --output=/data/%u/pathfm-full-evals/logs/%A_%a_cpu.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-7%8
#SBATCH --time=48:00:00
#SBATCH --partition=c
#SBATCH --account=sophont
#SBATCH --qos=high
#SBATCH --requeue

set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
[[ "${CPU_STAGE:-}" == "hest_probes" || "${CPU_STAGE:-}" == "pathorob_metrics" || "${CPU_STAGE:-}" == "thunder_cleanup" || "${CPU_STAGE:-}" == "cptac_pool" || "${CPU_STAGE:-}" == "cptac_probes" || "${CPU_STAGE:-}" == "thunder_summary" || "${CPU_STAGE:-}" == "hest_finalize" || "${CPU_STAGE:-}" == "pathorob_finalize" || "${CPU_STAGE:-}" == "cptac_finalize" ]]
[[ "${RETAIN_EMBEDDINGS:-}" == 0 || "${RETAIN_EMBEDDINGS:-}" == 1 ]]

repo=$PWD
tasks=()
case "$CPU_STAGE" in
  hest_probes)
    for index in 6 1 0 8 4 5 2 7 3; do
      tasks+=("hest probe $index")
    done
    ;;
  pathorob_metrics)
    for metric_dataset in \
      "clustering tcga" "apd tcga" "robustness tcga" \
      "clustering camelyon" "clustering tolkach_esca" \
      "apd camelyon" "apd tolkach_esca" \
      "robustness camelyon" "robustness tolkach_esca"; do
      tasks+=("pathorob metric $metric_dataset")
    done
    ;;
  thunder_cleanup)
    tasks=("thunder cleanup")
    ;;
  cptac_pool)
    tasks=("cptac pool")
    ;;
  cptac_probes)
    for index in 40 32 33 38 39 34 28 30 31 29 22 23 24 3 4 8 9 19 20 21 42 46 \
      41 0 1 2 10 11 12 13 14 16 17 18 15 48 47 49 35 36 37 25 26 27 43 44 \
      45 5 6 7; do
      tasks+=("cptac probe $index")
    done
    ;;
  thunder_summary)
    tasks=("thunder summary")
    ;;
  hest_finalize)
    tasks=("hest aggregate")
    ;;
  pathorob_finalize)
    tasks=("pathorob finalize")
    ;;
  cptac_finalize)
    tasks=("cptac finalize")
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
