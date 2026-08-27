# Current THUNDER website leaderboard: 16 classification datasets x KNN, linear
# probing/calibration, 16-shot SimpleShot, adversarial attack; four segmentation
# datasets with the official SegPath epoch overrides.
# https://mics-lab.github.io/thunder/leaderboards/

import atexit
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

from run_manifest import read_model_settings, verify_manifest

DATA_ROOT = "/data/thunder-data"
RUN_ROOT = f"/data/{os.environ['USER']}/pathfm-full-evals"
WORK_ROOT = f"{RUN_ROOT}/thunder"
Path(WORK_ROOT).mkdir(parents=True, exist_ok=True)
os.environ["THUNDER_BASE_DATA_FOLDER"] = WORK_ROOT
os.environ["THUNDER_DATA_FOLDER"] = DATA_ROOT
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["THUNDER_WANDB_MODE"] = "disabled"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from thunder import benchmark

assert len(sys.argv) >= 2
mode = sys.argv[1]
assert mode in (
    "precompute", "precompute_group", "cached_probe", "cached_group", "cleanup",
    "adversarial", "adversarial_group", "segmentation", "summary",
)
settings = read_model_settings()
MODEL_NAME = settings["MODEL_NAME"]
EXTRACTION_BATCH = settings["EXTRACTION_BATCH"]
ATTACK_BATCH = settings["ATTACK_BATCH"]
verify_manifest()
assert os.environ["RETAIN_EMBEDDINGS"] in ("0", "1")
RETAIN_EMBEDDINGS = os.environ["RETAIN_EMBEDDINGS"] == "1"
CLS_DATASETS = [
    "bach", "bracs", "break_his", "ccrcc", "crc", "esca", "mhist", "patch_camelyon",
    "spider_breast", "spider_colorectal", "spider_skin", "spider_thorax",
    "wilds", "tcga_crc_msi", "tcga_tils", "tcga_uniform",
]
SEG_DATASETS = ["pannuke", "ocelot", "segpath_epithelial", "segpath_lymphocytes"]
SEG_EPOCHS = {"segpath_epithelial": 9, "segpath_lymphocytes": 21}
CACHED_TASKS = ["linear_probing", "knn", "simple_shot"]
# Greedy balance by official split size (2,297,274 images total) and PGD sample cap.
PRECOMPUTE_GROUPS = [[12, 14, 10, 4, 11, 13, 3], [5, 7, 15, 8, 9, 1, 6, 2, 0]]
ADVERSARIAL_GROUPS = [[5, 8, 10, 12, 14, 4, 2, 0], [7, 9, 11, 13, 15, 3, 6, 1]]
FAST = True
MODEL_ID = MODEL_NAME + "_optimized"
RESULTS = f"{WORK_ROOT}/outputs/res"
EMBEDDINGS = f"{WORK_ROOT}/embeddings"
SEGMENTATION_EMBEDDINGS = (
    f"/tmp/{os.environ['USER']}/pathfm-full-evals/{os.environ['SLURM_JOB_ID']}/thunder"
    if mode == "segmentation" else "/tmp/pathfm-full-evals"
)
# Cap each extracted dense-feature tensor near the tested ViT-S batch footprint.
# This preserves large pooled-feature batches while safely accommodating models that
# return wider or denser segmentation tokens.
MAX_SEGMENTATION_BATCH_VALUES = 2048 * 256 * 384


if mode == "summary":
    assert len(sys.argv) == 2
    import pandas as pd
    from thunder.utils.results import gather_results

    gather_results()
    summary = Path(RESULTS, "results.csv")
    frame = pd.read_csv(summary)
    expected = {
        "benchmark_linear_probing", "benchmark_calibration", "benchmark_knn",
        "benchmark_simple_shot", "benchmark_adversarial_attack", "benchmark_segmentation",
    }
    aggregate = frame[(frame["model"] == MODEL_ID) & frame["dataset"].isin(expected)]
    assert len(aggregate) == len(expected) and set(aggregate["dataset"]) == expected
    assert aggregate["metric_score"].notna().all()
    print(aggregate[["dataset", "metric", "metric_score"]].to_string(index=False), flush=True)
    sys.exit(0)

if mode in ("precompute", "precompute_group", "adversarial", "adversarial_group", "segmentation"):
    from thunder.models.pretrained_models import PretrainedModel

    from model import EvalModel

    class ThunderModel(PretrainedModel):
        def __init__(self):
            super().__init__()
            self.encoder = EvalModel()
            self.name = self.encoder.name + "_optimized"
            assert self.name == MODEL_ID
            self.emb_dim = self.encoder.classification_dim
            self.emb_dim_seg = self.encoder.segmentation_dim
            self.vlm = False

        def get_transform(self):
            return self.encoder.transform(timm_style=True)

        def get_linear_probing_embeddings(self, images):
            with torch.autocast("cuda", torch.float16):
                return self.encoder.classification_features(images).float()

        def get_segmentation_embeddings(self, images):
            with torch.autocast("cuda", torch.float16):
                return self.encoder.segmentation_features(images).float()

        def forward(self, images):
            return self.get_linear_probing_embeddings(images)


class CachedThunderModel(torch.nn.Module):
    # Cached classification probes only require the model name and embedding width.
    # Avoid reloading a multi-GB backbone for every independently scheduled probe.
    def __init__(self, embedding_dim, segmentation_dim=None):
        super().__init__()
        self.name = MODEL_ID
        self.emb_dim = embedding_dim
        if segmentation_dim is not None:
            self.emb_dim_seg = segmentation_dim
        self.vlm = False

    def get_transform(self):
        raise RuntimeError("cached probes must not request an image transform")

    def get_linear_probing_embeddings(self, images):
        raise RuntimeError("cached probes must not run the backbone")

    def get_segmentation_embeddings(self, images):
        raise RuntimeError("cached probes must not run the backbone")


KWARGS = {
    "dataset.base_data_folder": DATA_ROOT,
    "task.pre_comp_emb_num_workers": 16,
}
if FAST:
    KWARGS.update({
        "task.pre_comp_emb_batch_size": EXTRACTION_BATCH,
        "task.attack_batch_size": ATTACK_BATCH,
        "adaptation.batch_size": 64,
    })
else:
    KWARGS["task.attack_batch_size"] = 32


if mode == "cleanup":
    assert len(sys.argv) == 2
    for dataset in CLS_DATASETS:
        expected = [
            f"{RESULTS}/{dataset}/{MODEL_ID}/{task}/frozen/outputs.json"
            for task in CACHED_TASKS
        ]
        assert all(os.path.exists(path) for path in expected), dataset
        for path in expected:
            json.loads(Path(path).read_text())
        cache = f"{EMBEDDINGS}/{dataset}/{MODEL_ID}"
        if os.path.exists(cache) and not RETAIN_EMBEDDINGS:
            shutil.rmtree(cache)
            print(f"[{MODEL_ID}/{dataset}] cached probes complete; cache deleted", flush=True)
        elif os.path.exists(cache):
            print(f"[{MODEL_ID}/{dataset}] cached probes complete; cache retained", flush=True)
    sys.exit(0)

assert len(sys.argv) >= 3 and sys.argv[2].isdigit()
index = int(sys.argv[2])
if mode == "precompute_group":
    assert len(sys.argv) == 3 and index < len(PRECOMPUTE_GROUPS)
    indices = PRECOMPUTE_GROUPS[index]
elif mode == "adversarial_group":
    assert len(sys.argv) == 3 and index < len(ADVERSARIAL_GROUPS)
    indices = ADVERSARIAL_GROUPS[index]
elif mode == "cached_group":
    assert len(sys.argv) == 3 and index < len(PRECOMPUTE_GROUPS)
    indices = PRECOMPUTE_GROUPS[index]
elif mode in ("precompute", "cached_probe", "adversarial"):
    assert index < len(CLS_DATASETS)
    indices = [index]
else:
    assert mode == "segmentation" and index < len(SEG_DATASETS)
    indices = [index]

if mode in ("precompute", "precompute_group"):
    model = ThunderModel()
    for index in indices:
        dataset = CLS_DATASETS[index]
        cache = f"{EMBEDDINGS}/{dataset}/{MODEL_ID}"
        expected = [
            f"{RESULTS}/{dataset}/{MODEL_ID}/{task}/frozen/outputs.json"
            for task in CACHED_TASKS
        ]
        if all(os.path.exists(path) for path in expected):
            for path in expected:
                json.loads(Path(path).read_text())
            if os.path.exists(cache) and not RETAIN_EMBEDDINGS:
                shutil.rmtree(cache)
            print(f"[{MODEL_ID}/{dataset}] cached probe outputs already complete", flush=True)
            continue
        if not os.path.exists(f"{cache}/.done"):
            if os.path.exists(cache):
                shutil.rmtree(cache)
            benchmark(model, dataset=dataset, task="pre_computing_embeddings", **KWARGS)
            Path(f"{cache}/.done").touch()
        Path(f"{cache}/.dim").write_text(f"{model.emb_dim}\n")
        print(f"[{MODEL_ID}/{dataset}] embeddings ready", flush=True)

elif mode in ("cached_probe", "cached_group"):
    assert len(sys.argv) == (4 if mode == "cached_probe" else 3)
    if mode == "cached_probe":
        assert sys.argv[3] in CACHED_TASKS
    for index in indices:
        dataset = CLS_DATASETS[index]
        cache = f"{EMBEDDINGS}/{dataset}/{MODEL_ID}"
        for task in ([sys.argv[3]] if mode == "cached_probe" else CACHED_TASKS):
            output = f"{RESULTS}/{dataset}/{MODEL_ID}/{task}/frozen/outputs.json"
            if os.path.exists(output):
                json.loads(Path(output).read_text())
                print(f"[{MODEL_ID}/{dataset}/{task}] already complete", flush=True)
                continue
            assert os.path.exists(f"{cache}/.done") and os.path.exists(f"{cache}/.dim")
            model = CachedThunderModel(int(Path(f"{cache}/.dim").read_text()))
            loading = "embedding_pre_loading" if task == "linear_probing" else "online_loading"
            benchmark(model, dataset=dataset, task=task, loading_mode=loading, **KWARGS)
            assert os.path.exists(output)
            json.loads(Path(output).read_text())
            print(f"[{MODEL_ID}/{dataset}/{task}] done", flush=True)

elif mode in ("adversarial", "adversarial_group"):
    model = ThunderModel()
    for index in indices:
        dataset = CLS_DATASETS[index]
        output = f"{RESULTS}/{dataset}/{MODEL_ID}/adversarial_attack/frozen/outputs.json"
        if os.path.exists(output):
            json.loads(Path(output).read_text())
            print(f"[{MODEL_ID}/{dataset}/adversarial_attack] already complete", flush=True)
            continue
        assert os.path.exists(
            f"{RESULTS}/{dataset}/{MODEL_ID}/linear_probing/frozen/outputs.json"
        )
        started = time.time()
        benchmark(
            model, dataset=dataset, task="adversarial_attack",
            loading_mode="online_loading", **KWARGS,
        )
        assert os.path.exists(output)
        json.loads(Path(output).read_text())
        print(
            f"[{MODEL_ID}/{dataset}/adversarial_attack] done in "
            f"{(time.time() - started) / 60:.1f} min",
            flush=True,
        )

else:
    assert mode == "segmentation" and len(sys.argv) == 3
    torch.set_float32_matmul_precision("high")
    dataset = SEG_DATASETS[index]
    cache = f"{SEGMENTATION_EMBEDDINGS}/{dataset}/{MODEL_ID}"
    legacy_cache = f"{EMBEDDINGS}/{dataset}/{MODEL_ID}"
    output = f"{RESULTS}/{dataset}/{MODEL_ID}/segmentation/frozen/outputs.json"
    if os.path.exists(output):
        json.loads(Path(output).read_text())
        if os.path.exists(SEGMENTATION_EMBEDDINGS):
            shutil.rmtree(SEGMENTATION_EMBEDDINGS)
        if os.path.exists(legacy_cache):
            shutil.rmtree(legacy_cache)
        print(f"[{MODEL_ID}/{dataset}/segmentation] already complete", flush=True)
        sys.exit(0)
    if os.path.exists(legacy_cache):
        shutil.rmtree(legacy_cache)
    cleanup_cache = lambda: (
        shutil.rmtree(SEGMENTATION_EMBEDDINGS)
        if os.path.exists(SEGMENTATION_EMBEDDINGS) else None
    )
    atexit.register(cleanup_cache)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(128 + signal.SIGINT))
    started = time.time()
    model = ThunderModel()
    segmentation_batch = min(
        EXTRACTION_BATCH,
        MAX_SEGMENTATION_BATCH_VALUES
        // (model.encoder.segmentation_tokens * model.emb_dim_seg),
    )
    assert segmentation_batch > 0
    if not os.path.exists(f"{cache}/.done"):
        if os.path.exists(cache):
            shutil.rmtree(cache)
        print(
            f"[{MODEL_ID}/{dataset}/segmentation] extraction batch="
            f"{segmentation_batch}",
            flush=True,
        )
        benchmark(
            model, dataset=dataset, task="pre_computing_embeddings",
            **{
                **KWARGS,
                "task.pre_comp_emb_batch_size": segmentation_batch,
                "task.base_embeddings_folder": SEGMENTATION_EMBEDDINGS,
            },
        )
        Path(f"{cache}/.done").touch()
    for split in ("train", "val", "test"):
        for filename in ("embeddings.h5", "labels.h5"):
            cached_split = f"{cache}/{split}/{filename}"
            assert os.path.exists(cached_split) and os.path.getsize(cached_split) > 0
    cached_model = CachedThunderModel(model.emb_dim, model.emb_dim_seg)
    del model
    torch.cuda.empty_cache()
    epochs = {"adaptation.epochs": SEG_EPOCHS[dataset]} if dataset in SEG_EPOCHS else {}
    benchmark(
        cached_model, dataset=dataset, task="segmentation", loading_mode="embedding_pre_loading",
        **{**KWARGS, **epochs, "adaptation.num_workers": 0,
           "task.base_embeddings_folder": SEGMENTATION_EMBEDDINGS},
    )
    assert os.path.exists(output)
    json.loads(Path(output).read_text())
    shutil.rmtree(SEGMENTATION_EMBEDDINGS)
    print(
        f"[{MODEL_ID}/{dataset}/segmentation] done in "
        f"{(time.time() - started) / 60:.1f} min; cache deleted",
        flush=True,
    )
