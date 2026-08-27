#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"
[[ $# == 1 && "$1" =~ ^(thunder|hest|cptac|pathorob)$ ]] || {
  echo "usage: ./submit_suite.sh {thunder|hest|cptac|pathorob}" >&2
  exit 2
}
mkdir -p "/data/$USER/pathfm-full-evals/logs"
preflight=$(sbatch --parsable --job-name=eval_preflight --array=0-0 --mem=32G \
  --export=ALL,EVAL_STAGE=preflight,PREFLIGHT_SUITES="$1" run_gpu.sbatch)

if [[ "$1" == "thunder" ]]; then
  extract=$(sbatch --parsable --job-name=thunder_precompute --array=0-1%2 \
    --dependency="afterok:$preflight" --export=ALL,EVAL_STAGE=thunder_precompute run_gpu.sbatch)
  probes=$(sbatch --parsable --job-name=thunder_cached --array=0-1%2 --mem=16G --dependency="afterok:$extract" \
    --export=ALL,EVAL_STAGE=thunder_cached run_gpu.sbatch)
  cleanup=$(sbatch --parsable --job-name=thunder_cleanup --array=0-0 \
    --dependency="afterok:$probes" --export=ALL,CPU_STAGE=thunder_cleanup run_cpu.sbatch)
  online=$(sbatch --parsable --job-name=thunder_online --array=0-5%2 --dependency="afterok:$cleanup" \
    --export=ALL,EVAL_STAGE=thunder_online run_gpu.sbatch)
  summary=$(sbatch --parsable --job-name=thunder_summary --array=0-0 \
    --dependency="afterok:$online" --export=ALL,CPU_STAGE=thunder_summary run_cpu.sbatch)
  echo "THUNDER preflight=$preflight precompute=$extract probes=$probes cleanup=$cleanup online=$online summary=$summary"
elif [[ "$1" == "hest" ]]; then
  extract=$(sbatch --parsable --job-name=hest_extract --array=0-2%2 --mem=80G \
    --dependency="afterok:$preflight" --export=ALL,EVAL_STAGE=hest_extract run_gpu.sbatch)
  probes=$(sbatch --parsable --job-name=hest_probes --array=0-8%8 --dependency="afterok:$extract" \
    --export=ALL,CPU_STAGE=hest_probes run_cpu.sbatch)
  final=$(sbatch --parsable --job-name=hest_finalize --array=0-0 \
    --dependency="afterok:$probes" --export=ALL,CPU_STAGE=hest_finalize run_cpu.sbatch)
  echo "HEST preflight=$preflight extract=$extract probes=$probes finalize=$final"
elif [[ "$1" == "pathorob" ]]; then
  extract=$(sbatch --parsable --job-name=pathorob_extract --array=0-2%2 --mem=144G \
    --dependency="afterok:$preflight" --export=ALL,EVAL_STAGE=pathorob_extract run_gpu.sbatch)
  metrics=$(sbatch --parsable --job-name=pathorob_metrics --array=0-8%8 --dependency="afterok:$extract" \
    --export=ALL,CPU_STAGE=pathorob_metrics run_cpu.sbatch)
  final=$(sbatch --parsable --job-name=pathorob_finalize --array=0-0 \
    --dependency="afterok:$metrics" --export=ALL,CPU_STAGE=pathorob_finalize run_cpu.sbatch)
  echo "PathoROB preflight=$preflight extract=$extract metrics=$metrics finalize=$final"
else
  extract=$(sbatch --parsable --job-name=cptac_extract --array=0-7%2 --mem=128G \
    --dependency="afterok:$preflight" --export=ALL,EVAL_STAGE=cptac_extract run_gpu.sbatch)
  pool=$(sbatch --parsable --job-name=cptac_pool --array=0-0 \
    --dependency="afterok:$extract" --export=ALL,CPU_STAGE=cptac_pool run_cpu.sbatch)
  probes=$(sbatch --parsable --job-name=cptac_probes --array=0-49%8 --dependency="afterok:$pool" \
    --export=ALL,CPU_STAGE=cptac_probes run_cpu.sbatch)
  final=$(sbatch --parsable --job-name=cptac_finalize --array=0-0 \
    --dependency="afterok:$probes" --export=ALL,CPU_STAGE=cptac_finalize run_cpu.sbatch)
  echo "CPTAC preflight=$preflight extract=$extract pool=$pool probes=$probes finalize=$final"
fi
