# Official Patho-Bench CPTAC tasks: 38 per-cohort classification tasks and four
# survival tasks, expanded to 50 independent classification/alpha fits, using
# case-level mean pooling and the benchmark's LogisticRegression/CoxNet experiments.
# https://github.com/mahmoodlab/Patho-Bench

import json
import os
import shutil
import sys
from pathlib import Path

from run_manifest import read_model_settings, verify_manifest

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
RUN_ROOT = f"/data/{os.environ['USER']}/pathfm-full-evals"

PATCH_DIR = "/data/Patho-Bench/job_dir/20x_p224_tp0.6/20x_224px_0px_overlap/patch_images"
SPLITS = "/data/Patho-Bench/splits"
assert len(sys.argv) >= 2 and sys.argv[1] in ("extract", "pool", "probe", "finalize")
settings = read_model_settings()
MODEL_NAME = settings["MODEL_NAME"]
verify_manifest()
assert os.environ["RETAIN_EMBEDDINGS"] in ("0", "1")
RETAIN_EMBEDDINGS = os.environ["RETAIN_EMBEDDINGS"] == "1"
OUT_ROOT = f"{RUN_ROOT}/cptac/{MODEL_NAME}"
DATASETS = [
    "cptac_brca", "cptac_ccrcc", "cptac_coad", "cptac_gbm", "cptac_hnsc",
    "cptac_lscc", "cptac_luad", "cptac_lung", "cptac_ov", "cptac_pda", "cptac_ucec",
]
PATCH_COUNT_MANIFEST = f"{RUN_ROOT}/cptac/patch_counts.tsv"
if os.path.exists(f"{OUT_ROOT}/.complete") and sys.argv[1] != "finalize":
    aggregate = json.loads(Path(OUT_ROOT, "aggregate.json").read_text())
    assert len(aggregate["classification_macro_ovr_auc"]) == 38
    assert len(aggregate["survival_cindex_by_alpha"]) == 4
    assert all(
        set(scores) == {"0.01", "0.02", "0.07"}
        for scores in aggregate["survival_cindex_by_alpha"].values()
    )
    print("DONE; all CPTAC results already complete", flush=True)
    sys.exit(0)

if sys.argv[1] == "extract":
    assert len(sys.argv) == 4 and sys.argv[2].isdigit() and sys.argv[3] == "8"
    worker = int(sys.argv[2])
    workers = int(sys.argv[3])
    assert worker < workers
    import h5py
    import pandas as pd
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, IterableDataset, get_worker_info

    from model import EXTRACTION_BATCH, EvalModel

    encoder = EvalModel()
    transform = encoder.transform(resize=False)

    class PatchStream(IterableDataset):
        def __init__(self, slide_ids):
            self.slide_ids = slide_ids

        def __iter__(self):
            worker_info = get_worker_info()
            worker_id = 0 if worker_info is None else worker_info.id
            workers = 1 if worker_info is None else worker_info.num_workers
            for slide_index in range(worker_id, len(self.slide_ids), workers):
                slide_dir = os.path.join(PATCH_DIR, self.slide_ids[slide_index])
                for name in sorted(os.listdir(slide_dir)):
                    if name.lower().endswith((".jpg", ".jpeg", ".png")):
                        with Image.open(os.path.join(slide_dir, name)) as handle:
                            image = handle.convert("RGB")
                        assert image.size == (224, 224)
                        yield transform(image), slide_index

    meanpool_dir = os.path.join(OUT_ROOT, "meanpool_embeddings")
    os.makedirs(meanpool_dir, exist_ok=True)
    required_slides = set()
    for dataset in DATASETS:
        for task in os.listdir(os.path.join(SPLITS, dataset)):
            required_slides.update(pd.read_csv(
                os.path.join(SPLITS, dataset, task, "k=all.tsv"),
                sep="\t", usecols=["slide_id"], dtype=str,
            )["slide_id"])
    available_slides = {
        name for name in os.listdir(PATCH_DIR)
        if os.path.isdir(os.path.join(PATCH_DIR, name))
    }
    assert required_slides <= available_slides
    assert len(required_slides) == 2083
    manifest = Path(PATCH_COUNT_MANIFEST)
    if not manifest.exists():
        manifest.parent.mkdir(parents=True, exist_ok=True)
        patch_counts = {
            slide_id: sum(
                name.lower().endswith((".jpg", ".jpeg", ".png"))
                for name in os.listdir(os.path.join(PATCH_DIR, slide_id))
            )
            for slide_id in sorted(required_slides)
        }
        temporary = manifest.with_suffix(f".{worker}.tmp")
        temporary.write_text("".join(
            f"{slide_id}\t{patch_counts[slide_id]}\n" for slide_id in sorted(patch_counts)
        ))
        os.replace(temporary, manifest)
    cached_patch_counts = {
        slide_id: int(count)
        for slide_id, count in (line.split("\t") for line in manifest.read_text().splitlines())
    }
    assert required_slides <= set(cached_patch_counts)
    patch_counts = {slide_id: cached_patch_counts[slide_id] for slide_id in required_slides}
    assert sum(patch_counts.values()) == 13_350_806
    slide_groups = [[] for _ in range(workers)]
    patch_totals = [0] * workers
    for slide_id in sorted(required_slides, key=lambda name: (-patch_counts[name], name)):
        shard = min(range(workers), key=lambda index: patch_totals[index])
        slide_groups[shard].append(slide_id)
        patch_totals[shard] += patch_counts[slide_id]
    slide_ids = slide_groups[worker]
    print(
        f"worker {worker}: {len(slide_ids)} slides, {patch_totals[worker]:,} patches; "
        f"eight-shard totals {patch_totals}",
        flush=True,
    )
    pending = []
    for slide_id in slide_ids:
        output = os.path.join(meanpool_dir, f"{slide_id}.h5")
        if not os.path.exists(output):
            pending.append(slide_id)
            continue
        with h5py.File(output) as handle:
            assert handle["features"].shape == (1, encoder.classification_dim), slide_id
            assert torch.isfinite(torch.from_numpy(handle["features"][:])).all(), slide_id
    if pending:
        pending_patches = sum(patch_counts[slide_id] for slide_id in pending)
        loader = DataLoader(
            PatchStream(pending), batch_size=EXTRACTION_BATCH, num_workers=16,
            pin_memory=True,
            prefetch_factor=4 if EXTRACTION_BATCH == 512 else 2,
        )
        feature_sums = torch.zeros(
            len(pending), encoder.classification_dim, dtype=torch.float64, device="cuda"
        )
        counts = torch.zeros(len(pending), dtype=torch.long, device="cuda")
        with torch.inference_mode():
            for batch_index, (images, slide_indices) in enumerate(loader):
                slide_indices = slide_indices.cuda(non_blocking=True)
                with torch.autocast("cuda", torch.float16):
                    features = encoder(images.cuda(non_blocking=True)).float()
                feature_sums.index_add_(0, slide_indices, features.double())
                counts.index_add_(0, slide_indices, torch.ones_like(slide_indices))
                if batch_index % 250 == 0:
                    processed = min((batch_index + 1) * EXTRACTION_BATCH, pending_patches)
                    print(
                        f"[worker {worker}] {processed:,}/{pending_patches:,} patches "
                        f"({processed / pending_patches:.1%})",
                        flush=True,
                    )
        feature_sums = feature_sums.cpu()
        counts = counts.cpu()
        assert torch.isfinite(feature_sums).all()
        assert torch.equal(counts, torch.tensor([patch_counts[slide_id] for slide_id in pending]))
        for slide_index, slide_id in enumerate(pending):
            output = os.path.join(meanpool_dir, f"{slide_id}.h5")
            temporary = output + ".tmp"
            with h5py.File(temporary, "w") as handle:
                handle.create_dataset(
                    "features",
                    data=(feature_sums[slide_index] / counts[slide_index]).float().numpy()[None],
                )
            os.replace(temporary, output)
    assert all(os.path.exists(os.path.join(meanpool_dir, f"{slide_id}.h5")) for slide_id in slide_ids)
    Path(f"{OUT_ROOT}/extract_{worker}_of_{workers}.done").touch()
    print(f"extract worker {worker} DONE", flush=True)

else:
    assert len(sys.argv) == (3 if sys.argv[1] == "probe" else 2)
    import h5py
    import numpy as np
    import pandas as pd
    import yaml

    tasks = []
    experiment_dir = os.path.join(OUT_ROOT, "experiment", "meanpool")
    for dataset in DATASETS:
        for task in sorted(os.listdir(os.path.join(SPLITS, dataset))):
            task_dir = os.path.join(SPLITS, dataset, task)
            split = os.path.join(task_dir, "k=all.tsv")
            config = os.path.join(task_dir, "config.yaml")
            with open(config) as handle:
                task_config = yaml.safe_load(handle)
            tasks.append((
                dataset, task, task_config["task_type"], task_config["sample_col"],
                split, config,
            ))
    assert sum(task[2] == "classification" for task in tasks) == 38
    assert sum(task[2] == "survival" for task in tasks) == 4
    assert all(task[3] == "case_id" for task in tasks)

    expected_results = []
    probe_tasks = []
    for dataset, task, task_type, sample_col, split, config in tasks:
        if task_type == "classification":
            expected_results.append((os.path.join(
                experiment_dir, dataset, task, "test_metrics_summary.json",
            ),))
            probe_tasks.append((dataset, task, task_type, sample_col, split, config, None))
        else:
            expected_results.extend((
                os.path.join(
                    experiment_dir, dataset, task, f"alpha_{alpha}",
                    "test_metrics_summary.json",
                ),
                os.path.join(
                    experiment_dir, dataset, task, f"alpha_{alpha}", "failed.json",
                ),
            ) for alpha in (0.01, 0.02, 0.07))
            probe_tasks.extend(
                (dataset, task, task_type, sample_col, split, config, alpha)
                for alpha in (0.01, 0.02, 0.07)
            )
    assert len(probe_tasks) == 50

    meanpool_dir = os.path.join(OUT_ROOT, "meanpool_embeddings")
    case_dir = os.path.join(OUT_ROOT, "case_meanpool_embeddings")
    assert all(os.path.exists(f"{OUT_ROOT}/extract_{shard}_of_8.done") for shard in range(8))
    if sys.argv[1] == "pool":
        case2slides = {}
        for _, _, _, sample_col, split, _ in tasks:
            if sample_col == "slide_id":
                continue
            frame = pd.read_csv(split, sep="\t", dtype=str)
            for case_id, group in frame.groupby("case_id"):
                case2slides.setdefault(case_id, set()).update(group["slide_id"])
        os.makedirs(case_dir, exist_ok=True)
        print(f"{len(case2slides)} cases", flush=True)
        for case_id, slide_ids in case2slides.items():
            output = os.path.join(case_dir, f"{case_id}.h5")
            if os.path.exists(output):
                with h5py.File(output) as handle:
                    assert handle["features"].ndim == 2 and handle["features"].shape[0] == 1
                    assert np.isfinite(handle["features"][:]).all()
                continue
            vectors = []
            for slide_id in sorted(slide_ids):
                with h5py.File(os.path.join(meanpool_dir, f"{slide_id}.h5")) as handle:
                    vector = handle["features"][:]
                    assert vector.ndim == 2 and vector.shape[0] == 1, slide_id
                    assert np.isfinite(vector).all(), slide_id
                    vectors.append(vector.astype(np.float64))
            embedding = np.concatenate(vectors).mean(0).astype(np.float32)[None, :]
            assert np.isfinite(embedding).all()
            temporary = output + ".tmp"
            with h5py.File(temporary, "w") as handle:
                handle.create_dataset("features", data=embedding)
            os.replace(temporary, output)
        Path(f"{OUT_ROOT}/.pooled").touch()
        print("case pooling DONE", flush=True)

    elif sys.argv[1] == "probe":
        assert os.path.exists(f"{OUT_ROOT}/.pooled")
        index = int(sys.argv[2])
        assert index < len(probe_tasks)
        dataset, task, task_type, sample_col, split, config, alpha = probe_tasks[index]
        output = os.path.join(experiment_dir, dataset, task)
        from patho_bench.ExperimentFactory import ExperimentFactory

        pooled_dir = meanpool_dir if sample_col == "slide_id" else case_dir

        if task_type == "classification":
            assert alpha is None
            result = os.path.join(output, "test_metrics_summary.json")
            if not os.path.exists(result):
                experiment = ExperimentFactory.linprobe(
                    split=split, task_config=config, pooled_embeddings_dir=pooled_dir,
                    saveto=output, combine_slides_per_patient=False,
                    cost=0.5, balanced=True, num_bootstraps=100,
                )
                experiment.train()
                experiment.test()
                auc = experiment.report_results(metric="macro-ovr-auc")
                print(f"{dataset}--{task}: macro-ovr-auc {auc:.4f}", flush=True)
            json.loads(Path(result).read_text())
        else:
            assert alpha is not None
            alpha_output = os.path.join(output, f"alpha_{alpha}")
            result = os.path.join(alpha_output, "test_metrics_summary.json")
            failed = os.path.join(alpha_output, "failed.json")
            if not os.path.exists(result) and not os.path.exists(failed):
                experiment = ExperimentFactory.coxnet(
                    split=split, task_config=config, pooled_embeddings_dir=pooled_dir,
                    saveto=alpha_output, combine_slides_per_patient=False,
                    alpha=alpha, l1_ratio=0.5, num_bootstraps=100,
                )
                experiment.train()
                if len(experiment.models) == experiment.dataset.num_folds:
                    assert experiment.train_error is None
                    experiment.test()
                if not os.path.exists(result):
                    assert "Numerical error, because weights are too large" in experiment.train_error
                    Path(failed).write_text(json.dumps({
                        "status": "upstream_numerical_failure",
                        "trained_folds": len(experiment.models),
                        "expected_folds": experiment.dataset.num_folds,
                        "message": experiment.train_error,
                    }, indent=2) + "\n")
            if os.path.exists(result):
                json.loads(Path(result).read_text())
                print(f"{dataset}--{task}: coxnet alpha={alpha} done", flush=True)
            else:
                failure = json.loads(Path(failed).read_text())
                assert failure["status"] == "upstream_numerical_failure"
                print(f"{dataset}--{task}: coxnet alpha={alpha} numerical failure", flush=True)

    else:
        assert sys.argv[1] == "finalize"
        assert all(sum(os.path.exists(path) for path in paths) == 1 for paths in expected_results)
        for paths in expected_results:
            json.loads(Path(next(path for path in paths if os.path.exists(path))).read_text())
        classification = {}
        survival = {}
        for dataset, task, task_type, _, _, _ in tasks:
            key = f"{dataset}/{task}"
            if task_type == "classification":
                result = json.loads(Path(
                    experiment_dir, dataset, task, "test_metrics_summary.json",
                ).read_text())
                classification[key] = float(result["macro-ovr-auc"]["mean"])
            else:
                survival[key] = {}
                for alpha in (0.01, 0.02, 0.07):
                    result_path = Path(
                        experiment_dir, dataset, task, f"alpha_{alpha}",
                        "test_metrics_summary.json",
                    )
                    survival[key][str(alpha)] = (
                        float(json.loads(result_path.read_text())["cindex"]["mean"])
                        if result_path.exists() else None
                    )
        aggregate = Path(OUT_ROOT, "aggregate.json")
        temporary = aggregate.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "classification_macro_ovr_auc": classification,
            "mean_classification_macro_ovr_auc": sum(classification.values()) / len(classification),
            "survival_cindex_by_alpha": survival,
        }, indent=2) + "\n")
        os.replace(temporary, aggregate)
        if not RETAIN_EMBEDDINGS:
            if os.path.exists(meanpool_dir):
                shutil.rmtree(meanpool_dir)
            if os.path.exists(case_dir):
                shutil.rmtree(case_dir)
        Path(f"{OUT_ROOT}/.complete").touch()
        print(
            f"probes DONE; temporary embeddings {'retained' if RETAIN_EMBEDDINGS else 'deleted'}",
            flush=True,
        )
