# Full official PathoROB evaluation: robustness index, average performance drop,
# and clustering score on camelyon, tcga, and tolkach_esca.
# https://github.com/bifold-pathomics/PathoROB

import atexit
import json
import os
import shutil
import signal
import sys
from pathlib import Path

from run_manifest import read_model_settings, verify_manifest

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/data/nanopath_full_evals/huggingface"
RUN_ROOT = f"/data/{os.environ['USER']}/pathfm-full-evals"
PATHOROB_BATCH = 256

assert len(sys.argv) >= 2 and sys.argv[1] in ("extract", "metric", "finalize")
settings = read_model_settings()
MODEL_NAME = settings["MODEL_NAME"]
verify_manifest()
if sys.argv[1] == "extract":
    assert len(sys.argv) == 3 and sys.argv[2] in ("camelyon", "tcga", "tolkach_esca")
    scratch_root = Path(
        f"/tmp/{os.environ['USER']}/pathfm-full-evals/{os.environ['SLURM_JOB_ID']}"
    )
    os.environ["HF_DATASETS_CACHE"] = f"{scratch_root}/pathorob/{sys.argv[2]}"
    cleanup_scratch = lambda: shutil.rmtree(scratch_root) if scratch_root.exists() else None
    atexit.register(cleanup_scratch)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(128 + signal.SIGINT))
    from model import AMP_DTYPE, EvalModel


METADATA_DIR = "/data/nanopath_full_evals/repos/PathoROB/data/metadata"
DATA_ROOT = "/data/pathorob"
FEATURES_DIR = f"{RUN_ROOT}/pathorob/features"
RESULTS_ROOT = f"{RUN_ROOT}/pathorob/results"
NAME = f"{MODEL_NAME}_clsmean"
DATASETS = ["camelyon", "tcga", "tolkach_esca"]
if sys.argv[1] == "extract":
    dataset = sys.argv[2]
EXPECTED = {
    dataset: [
        f"{RESULTS_ROOT}/robustness_index/{NAME}/{dataset}/-1_0/results_summary.json",
        f"{RESULTS_ROOT}/apd/{NAME}/{dataset}_summary.json",
        f"{RESULTS_ROOT}/apd/{NAME}/{dataset}_raw.json",
        f"{RESULTS_ROOT}/clustering_score/{NAME}/{dataset}/results_summary.json",
    ]
    for dataset in DATASETS
}

if sys.argv[1] == "extract":
    import torch
    from pathorob.features.data_manager import FeatureDataManager
    from pathorob.features.extract_features import extract_features
    from pathorob.models.utils import ModelWrapper

    class Wrapper(ModelWrapper):
        def __init__(self):
            self.encoder = EvalModel()

        def get_model(self):
            return self.encoder

        def get_preprocess(self):
            return self.encoder.transform()

        @torch.no_grad()
        def extract(self, data):
            with torch.autocast(
                "cuda", getattr(torch, AMP_DTYPE), enabled=AMP_DTYPE != "float32",
            ):
                return self.encoder.clsmean_features(data).float()

    dataset_dir = os.path.join(FEATURES_DIR, NAME, dataset)
    marker = os.path.join(dataset_dir, ".done")
    if all(os.path.exists(path) for path in EXPECTED[dataset]) and os.path.exists(
        f"{RESULTS_ROOT}/apd/{NAME}/aggregated_summary.json"
    ):
        for path in EXPECTED[dataset]:
            json.loads(Path(path).read_text())
        json.loads(Path(f"{RESULTS_ROOT}/apd/{NAME}/aggregated_summary.json").read_text())
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        print(f"DONE PathoROB-{dataset}; results already complete", flush=True)
        sys.exit(0)
    if not os.path.exists(marker):
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        wrapper = Wrapper()
        assert wrapper.encoder.name == MODEL_NAME
        extract_features(
            model_wrapper=wrapper,
            data_manager=FeatureDataManager(features_dir=FEATURES_DIR, metadata_dir=METADATA_DIR),
            model_name=NAME,
            dataset_name=dataset,
            dataset_path=f"{DATA_ROOT}/{dataset}",
            batch_size=PATHOROB_BATCH,
            num_workers=16,
            device="cuda",
        )
        cleanup_scratch()
        atexit.unregister(cleanup_scratch)
        Path(marker).touch()
    print(f"DONE PathoROB-{dataset} feature extraction", flush=True)

else:
    assert sys.argv[1] in ("metric", "finalize")
    complete = all(os.path.exists(path) for paths in EXPECTED.values() for path in paths)
    complete = complete and os.path.exists(f"{RESULTS_ROOT}/apd/{NAME}/aggregated_summary.json")
    if complete:
        for paths in EXPECTED.values():
            for path in paths:
                json.loads(Path(path).read_text())
        json.loads(Path(f"{RESULTS_ROOT}/apd/{NAME}/aggregated_summary.json").read_text())
        stale_features = os.path.join(FEATURES_DIR, NAME)
        if os.path.exists(stale_features):
            shutil.rmtree(stale_features)
        print("DONE; all PathoROB results already exist and temporary features are absent", flush=True)
        sys.exit(0)

    for dataset in DATASETS:
        assert os.path.exists(os.path.join(FEATURES_DIR, NAME, dataset, ".done")), dataset
    if sys.argv[1] == "metric":
        assert len(sys.argv) >= 3 and sys.argv[2] in ("robustness", "apd", "clustering")
        if sys.argv[2] == "robustness":
            from pathorob.robustness_index.robustness_index import (
                compute as compute_robustness,
            )

            assert len(sys.argv) == 4 and sys.argv[3] in DATASETS
            dataset = sys.argv[3]
            robustness_result = f"{RESULTS_ROOT}/robustness_index/{NAME}/{dataset}/-1_0/results_summary.json"
            if os.path.exists(robustness_result):
                json.loads(Path(robustness_result).read_text())
                sys.exit(0)
            compute_robustness(
                model=NAME,
                dataset=dataset,
                features_dir=FEATURES_DIR,
                metadata_dir=METADATA_DIR,
                results_dir=f"{RESULTS_ROOT}/robustness_index",
                figures_subdir=f"{RESULTS_ROOT}/robustness_index/fig",
                num_workers=8,
                plot_graphs=False,
            )
        elif sys.argv[2] == "apd":
            from pathorob.apd.apd import compute as compute_apd

            assert len(sys.argv) == 4 and sys.argv[3] in DATASETS
            if os.path.exists(f"{RESULTS_ROOT}/apd/{NAME}/{sys.argv[3]}_summary.json") and os.path.exists(
                f"{RESULTS_ROOT}/apd/{NAME}/{sys.argv[3]}_raw.json"
            ):
                json.loads(Path(f"{RESULTS_ROOT}/apd/{NAME}/{sys.argv[3]}_summary.json").read_text())
                json.loads(Path(f"{RESULTS_ROOT}/apd/{NAME}/{sys.argv[3]}_raw.json").read_text())
                sys.exit(0)
            compute_apd(
                model=NAME, dataset=sys.argv[3],
                features_dir=FEATURES_DIR,
                metadata_dir=METADATA_DIR,
                results_dir=f"{RESULTS_ROOT}/apd",
                iterations=20,
            )
        else:
            from pathorob.clustering_score.clustering_score import (
                compute as compute_clustering,
            )

            assert len(sys.argv) == 4 and sys.argv[3] in DATASETS
            compute_clustering(
                model=NAME, dataset=sys.argv[3],
                features_dir=FEATURES_DIR,
                metadata_dir=METADATA_DIR,
                results_dir=f"{RESULTS_ROOT}/clustering_score",
                K=None,
                minK=2,
                maxK=30,
                metric="cosine",
                num_trials=50,
                overwrite_results=False,
            )
    else:
        from pathorob.apd.apd import compute_all as compute_all_apd

        assert len(sys.argv) == 2
        compute_all_apd({
            "model": NAME, "datasets": list(DATASETS), "features_dir": FEATURES_DIR,
            "metadata_dir": METADATA_DIR, "results_dir": f"{RESULTS_ROOT}/apd",
            "iterations": 20, "overwrite_results": False,
        })
        for dataset in DATASETS:
            assert all(os.path.exists(path) for path in EXPECTED[dataset]), dataset
        assert os.path.exists(f"{RESULTS_ROOT}/apd/{NAME}/aggregated_summary.json")
        for paths in EXPECTED.values():
            for path in paths:
                json.loads(Path(path).read_text())
        json.loads(Path(f"{RESULTS_ROOT}/apd/{NAME}/aggregated_summary.json").read_text())
        shutil.rmtree(os.path.join(FEATURES_DIR, NAME))
        print("DONE; all three PathoROB metrics saved and temporary features deleted", flush=True)
