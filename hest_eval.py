# Current official HEST-bench: nine tasks, PCA-256 Ridge regression, Pearson correlation.
# https://github.com/mahmoodlab/HEST/tree/main/bench

import atexit
import json
import os
import shutil
import signal
import sys
from pathlib import Path

from run_manifest import read_model_settings, verify_manifest

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["HF_HUB_OFFLINE"] = "1"

BENCH_DATA_ROOT = "/data/HEST"
RUN_ROOT = f"/data/{os.environ['USER']}/pathfm-full-evals"
RESULTS_ROOT = f"{RUN_ROOT}/hest/results"
EMBED_ROOT = f"{RUN_ROOT}/hest/embeddings"
DATASETS = ["IDC", "PRAD", "PAAD", "SKCM", "COAD", "READ", "CCRCC", "LUNG", "LYMPH_IDC"]
GROUPS = [[6, 3], [1, 5, 2], [0, 8, 4, 7]]  # 77,254 / 78,688 / 76,357 spots

assert len(sys.argv) >= 2 and sys.argv[1] in ("extract", "extract_group", "probe", "aggregate")
settings = read_model_settings()
MODEL_NAME = settings["MODEL_NAME"]
verify_manifest()
assert os.environ["RETAIN_EMBEDDINGS"] in ("0", "1")
RETAIN_EMBEDDINGS = os.environ["RETAIN_EMBEDDINGS"] == "1"

if sys.argv[1] in ("extract", "extract_group"):
    assert len(sys.argv) == 3
    if sys.argv[1] == "extract_group":
        assert sys.argv[2] in ("0", "1", "2")
    else:
        assert sys.argv[2].isdigit()
    indices = GROUPS[int(sys.argv[2])] if sys.argv[1] == "extract_group" else [int(sys.argv[2])]
    assert all(index < len(DATASETS) for index in indices)

    import h5py
    import pandas as pd
    import torch
    from hest.bench.benchmark import embed_tiles
    from hest.bench.st_dataset import H5PatchDataset
    from torch.utils.data import ConcatDataset, DataLoader

    from model import EXTRACTION_BATCH, EvalModel

    encoder = EvalModel()
    assert encoder.name == MODEL_NAME
    transform = encoder.transform(resize=False)
    scratch_root = Path(
        f"/tmp/{os.environ['USER']}/pathfm-full-evals/{os.environ['SLURM_JOB_ID']}/hest"
    )
    cleanup_scratch = lambda: shutil.rmtree(scratch_root) if scratch_root.exists() else None
    atexit.register(cleanup_scratch)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(128 + signal.SIGINT))
    for index in indices:
        dataset = DATASETS[index]
        summary = Path(f"{RESULTS_ROOT}/{MODEL_NAME}/{dataset}/summary.json")
        embedding_dir = Path(f"{EMBED_ROOT}/{MODEL_NAME}/{dataset}")
        if summary.exists():
            float(json.loads(summary.read_text())["pearson"])
            for run_dir in summary.parent.glob(f"{MODEL_NAME}::*"):
                shutil.rmtree(run_dir)
            if embedding_dir.exists() and not RETAIN_EMBEDDINGS:
                shutil.rmtree(embedding_dir)
            print(f"DONE {dataset}; result already exists", flush=True)
            continue
        samples = pd.concat([
            pd.read_csv(f"{BENCH_DATA_ROOT}/{dataset}/splits/train_0.csv"),
            pd.read_csv(f"{BENCH_DATA_ROOT}/{dataset}/splits/test_0.csv"),
        ]).drop_duplicates("sample_id")
        target = embedding_dir / "custom_encoder"
        target.mkdir(parents=True, exist_ok=True)
        local_target = scratch_root / MODEL_NAME / dataset / "custom_encoder"
        local_target.mkdir(parents=True, exist_ok=True)
        pending = [row for row in samples.itertuples(index=False) if not (target / f"{row.sample_id}.h5").exists()]
        if pending:
            patch_datasets = [
                H5PatchDataset(
                    f"{BENCH_DATA_ROOT}/{dataset}/{row.patches_path}", img_transform=transform,
                )
                for row in pending
            ]
            temporary = local_target / "combined.h5"
            if temporary.exists():
                temporary.unlink()
            embed_tiles(
                DataLoader(
                    ConcatDataset(patch_datasets), batch_size=EXTRACTION_BATCH,
                    num_workers=16, shuffle=False,
                    pin_memory=True,
                    prefetch_factor=4 if EXTRACTION_BATCH == 512 else 2,
                ),
                encoder, str(temporary), "cuda", torch.float16,
            )
            with h5py.File(temporary) as combined:
                offset = 0
                for row, patch_dataset in zip(pending, patch_datasets):
                    output = target / f"{row.sample_id}.h5"
                    publishing = output.with_suffix(".publishing")
                    if publishing.exists():
                        publishing.unlink()
                    stop = offset + len(patch_dataset)
                    with h5py.File(publishing, "w") as sample_output:
                        for key, values in combined.items():
                            sample_output.create_dataset(
                                key, data=values[offset:stop], dtype=values.dtype,
                            )
                    os.replace(publishing, output)
                    offset = stop
                assert offset == len(combined["embeddings"])
            temporary.unlink()
        (embedding_dir / ".done").touch()
        print(f"DONE {dataset} feature extraction", flush=True)
    cleanup_scratch()
    atexit.unregister(cleanup_scratch)

elif sys.argv[1] == "probe":
    assert len(sys.argv) == 3 and sys.argv[2].isdigit()
    index = int(sys.argv[2])
    assert index < len(DATASETS)
    dataset = DATASETS[index]
    summary = Path(f"{RESULTS_ROOT}/{MODEL_NAME}/{dataset}/summary.json")
    embedding_dir = Path(f"{EMBED_ROOT}/{MODEL_NAME}/{dataset}")
    if summary.exists():
        float(json.loads(summary.read_text())["pearson"])
        for run_dir in summary.parent.glob(f"{MODEL_NAME}::*"):
            shutil.rmtree(run_dir)
        if embedding_dir.exists() and not RETAIN_EMBEDDINGS:
            shutil.rmtree(embedding_dir)
        print(f"DONE {dataset}; result already exists", flush=True)
        sys.exit(0)
    assert (embedding_dir / ".done").exists()

    import torch
    from hest.bench import benchmark

    _, performance = benchmark(
        torch.nn.Identity(),
        torch.nn.Identity(),
        torch.float16,
        batch_size=512,
        num_workers=16,
        bench_data_root=BENCH_DATA_ROOT,
        results_dir=f"{RESULTS_ROOT}/{MODEL_NAME}/{dataset}",
        embed_dataroot=f"{EMBED_ROOT}/{MODEL_NAME}",
        exp_code=MODEL_NAME,
        datasets=[dataset],
        encoders=[],
    )
    score = float(performance["custom_encoder"])
    summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary.with_suffix(".tmp")
    temporary.write_text(json.dumps({"dataset": dataset, "pearson": score}, indent=2) + "\n")
    os.replace(temporary, summary)
    for run_dir in summary.parent.glob(f"{MODEL_NAME}::*"):
        shutil.rmtree(run_dir)
    if RETAIN_EMBEDDINGS:
        print(f"DONE {dataset}: Pearson {score:.4f}; embeddings retained", flush=True)
    else:
        shutil.rmtree(embedding_dir)
        print(f"DONE {dataset}: Pearson {score:.4f}; embeddings deleted", flush=True)

else:
    assert sys.argv[1] == "aggregate" and len(sys.argv) == 2
    scores = {}
    for dataset in DATASETS:
        summary = Path(f"{RESULTS_ROOT}/{MODEL_NAME}/{dataset}/summary.json")
        assert summary.exists(), dataset
        scores[dataset] = float(json.loads(summary.read_text())["pearson"])
    aggregate = Path(f"{RESULTS_ROOT}/{MODEL_NAME}/aggregate.json")
    temporary = aggregate.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "datasets": scores,
        "mean_pearson": sum(scores.values()) / len(scores),
    }, indent=2) + "\n")
    os.replace(temporary, aggregate)
    if not RETAIN_EMBEDDINGS:
        embedding_root = Path(f"{EMBED_ROOT}/{MODEL_NAME}")
        if embedding_root.exists():
            assert not any(embedding_root.iterdir())
            embedding_root.rmdir()
    print(f"DONE: mean Pearson {sum(scores.values()) / len(scores):.4f}", flush=True)
