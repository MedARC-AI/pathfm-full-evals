#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"
[[ $# == 0 || ( $# == 1 && "$1" == "--no-pathorob" ) ]] || {
  echo "usage: ./submit_all.sh [--no-pathorob]" >&2
  exit 2
}
mkdir -p "/data/$USER/pathfm-full-evals/logs"

preflight_suites=thunder:hest:cptac:pathorob
[[ $# == 0 ]] || preflight_suites=thunder:hest:cptac
preflight_job=$(sbatch --parsable --job-name=eval_preflight --array=0-0 --mem=32G \
  --export=ALL,EVAL_STAGE=preflight,PREFLIGHT_SUITES="$preflight_suites" run_gpu.sbatch)
if [[ $# == 0 ]]; then
  extract_job=$(sbatch --parsable --job-name=hest_pathorob --array=0-5%2 --mem=144G \
    --dependency="afterok:$preflight_job" --export=ALL,EVAL_STAGE=hest_pathorob run_gpu.sbatch)
else
  extract_job=$(sbatch --parsable --job-name=hest_extract --array=0-2%2 --mem=80G \
    --dependency="afterok:$preflight_job" --export=ALL,EVAL_STAGE=hest_extract run_gpu.sbatch)
fi
hest_probe_job=$(sbatch --parsable --job-name=hest_probes --array=0-8%8 \
  --dependency="afterok:$extract_job" --export=ALL,CPU_STAGE=hest_probes run_cpu.sbatch)
hest_final_job=$(sbatch --parsable --job-name=hest_finalize --array=0-0 \
  --dependency="afterok:$hest_probe_job" --export=ALL,CPU_STAGE=hest_finalize run_cpu.sbatch)
if [[ $# == 0 ]]; then
  pathorob_metrics_job=$(sbatch --parsable --job-name=pathorob_metrics --array=0-8%8 \
    --dependency="afterok:$extract_job" --export=ALL,CPU_STAGE=pathorob_metrics run_cpu.sbatch)
  pathorob_final_job=$(sbatch --parsable --job-name=pathorob_finalize --array=0-0 \
    --dependency="afterok:$pathorob_metrics_job" --export=ALL,CPU_STAGE=pathorob_finalize run_cpu.sbatch)
fi
precompute_job=$(sbatch --parsable --job-name=thunder_precompute --array=0-1%2 \
  --dependency="afterok:$extract_job" --export=ALL,EVAL_STAGE=thunder_precompute run_gpu.sbatch)
cached_job=$(sbatch --parsable --job-name=thunder_cached --array=0-1%2 --mem=16G \
  --dependency="afterok:$precompute_job" --export=ALL,EVAL_STAGE=thunder_cached run_gpu.sbatch)
cleanup_job=$(sbatch --parsable --job-name=thunder_cleanup \
  --array=0-0 --dependency="afterok:$cached_job" --export=ALL,CPU_STAGE=thunder_cleanup run_cpu.sbatch)
cptac_extract_job=$(sbatch --parsable --job-name=cptac_extract --array=0-7%2 --mem=128G \
  --dependency="afterok:$cleanup_job" --export=ALL,EVAL_STAGE=cptac_extract run_gpu.sbatch)
thunder_job=$(sbatch --parsable --job-name=thunder_online --array=0-5%2 \
  --dependency="afterok:$cptac_extract_job" --export=ALL,EVAL_STAGE=thunder_online run_gpu.sbatch)
pool_job=$(sbatch --parsable --job-name=cptac_pool \
  --array=0-0 --dependency="afterok:$cptac_extract_job" --export=ALL,CPU_STAGE=cptac_pool run_cpu.sbatch)
cptac_job=$(sbatch --parsable --job-name=cptac_probes --array=0-49%8 \
  --dependency="afterok:$pool_job" --export=ALL,CPU_STAGE=cptac_probes run_cpu.sbatch)
cptac_final_job=$(sbatch --parsable --job-name=cptac_finalize --array=0-0 \
  --dependency="afterok:$cptac_job" --export=ALL,CPU_STAGE=cptac_finalize run_cpu.sbatch)
summary_job=$(sbatch --parsable --job-name=thunder_summary \
  --array=0-0 --dependency="afterok:$thunder_job" --export=ALL,CPU_STAGE=thunder_summary run_cpu.sbatch)
if [[ $# == 0 ]]; then
  echo "preflight=$preflight_job HEST-PathoROB=$extract_job HEST-probes=$hest_probe_job HEST-finalize=$hest_final_job PathoROB-metrics=$pathorob_metrics_job PathoROB-finalize=$pathorob_final_job THUNDER-precompute=$precompute_job THUNDER-cached-probes=$cached_job THUNDER-cleanup=$cleanup_job CPTAC-extract=$cptac_extract_job CPTAC-pool=$pool_job CPTAC-probes=$cptac_job CPTAC-finalize=$cptac_final_job THUNDER-online=$thunder_job THUNDER-summary=$summary_job"
else
  echo "preflight=$preflight_job HEST-extract=$extract_job HEST-probes=$hest_probe_job HEST-finalize=$hest_final_job THUNDER-precompute=$precompute_job THUNDER-cached-probes=$cached_job THUNDER-cleanup=$cleanup_job CPTAC-extract=$cptac_extract_job CPTAC-pool=$pool_job CPTAC-probes=$cptac_job CPTAC-finalize=$cptac_final_job THUNDER-online=$thunder_job THUNDER-summary=$summary_job"
fi
