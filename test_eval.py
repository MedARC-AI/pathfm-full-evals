import tempfile
from pathlib import Path

import numpy as np
import torch
from thunder import benchmark, download_datasets, download_models, generate_splits
from thunder.tasks.knn_classification import knn_predict

from run_manifest import build_manifest, verify_manifest, write_or_verify_manifest

assert callable(benchmark) and callable(download_datasets)
assert callable(download_models) and callable(generate_splits)

rng = np.random.default_rng(1337)
train = rng.normal(size=(97, 31)).astype(np.float32)
query = rng.normal(size=(23, 31)).astype(np.float32)
labels = rng.integers(0, 5, size=len(train), dtype=np.int64)
predictions = knn_predict(train, labels, query, [1, 5, 17])
train_normalized = train / np.linalg.norm(train, axis=1, keepdims=True)
query_normalized = query / np.linalg.norm(query, axis=1, keepdims=True)
neighbors = np.argsort(-(query_normalized @ train_normalized.T), axis=1)
for k in (1, 5, 17):
    expected = np.array([
        np.bincount(labels[row[:k]], minlength=5).argmax() for row in neighbors
    ])
    assert np.array_equal(predictions[k], expected)

features = torch.randn(4, 256, 384)
scales = features.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-12).div_(127)
quantized = torch.clamp(torch.round(features / scales), -127, 127).to(torch.int8)
restored = quantized.float() * scales
cosine = torch.nn.functional.cosine_similarity(features.flatten(), restored.flatten(), dim=0)
assert cosine > 0.9999
assert (features - restored).abs().max() <= scales.max() / 2 + 1e-6

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    asset = root / "checkpoint.pt"
    asset.write_bytes(b"checkpoint-v1")
    model_path = root / "model.py"
    source = f'''\
MODEL_NAME = "synthetic-model"
EXTRACTION_BATCH = 64
ATTACK_BATCH = 16
MODEL_ASSETS = ["{asset}"]
'''
    model_path.write_text(source)
    first = build_manifest(model_path)
    model_path.write_text(source.replace("EXTRACTION_BATCH = 64", "EXTRACTION_BATCH = 128"))
    assert build_manifest(model_path) == first
    model_path.write_text(source)
    target = write_or_verify_manifest(model_path, root / "run")
    assert target.exists() and verify_manifest(model_path, root / "run") == target
    asset.write_bytes(b"checkpoint-v2")
    model_path.write_text(source)
    assert build_manifest(model_path) != first

for name in ("hest_eval.py", "pathorob_eval.py", "thunder_eval.py"):
    source = (Path(__file__).parent / name).read_text()
    assert "SLURM_JOB_ID" in source and "os.environ['USER']" in source
for name in ("run_gpu.sbatch", "run_cpu.sbatch"):
    source = (Path(__file__).parent / name).read_text()
    assert "noclobber" not in source and "scontrol requeue" not in source
    assert "RETAIN_EMBEDDINGS" in source
for name in ("submit_all.sh", "submit_suite.sh"):
    source = (Path(__file__).parent / name).read_text()
    assert "--embeddings=remove" in source and "--embeddings=retain" in source
for name in ("hest_eval.py", "pathorob_eval.py", "thunder_eval.py", "cptac_eval.py"):
    assert "RETAIN_EMBEDDINGS" in (Path(__file__).parent / name).read_text()

print("PASS manifest, THUNDER API, KNN, int8 cache, scratch isolation, retention, and native arrays")
