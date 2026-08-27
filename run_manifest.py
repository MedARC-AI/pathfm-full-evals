import ast
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

PROTOCOL_VERSION = "pathfm-optimized-v1"
REPO_ROOT = Path(__file__).parent
RUN_ROOT = Path(f"/data/{os.environ['USER']}/pathfm-full-evals")
BENCHMARKS = {
    "thunder": "3d1cc9513fb2cfd8c4afb0d7bb9f5c4f6b69117f",
    "hest": "3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3",
    "patho_bench": "660e77044640e3d7d2f1150cc6721e97454993bf",
    "pathorob": "6583cf0b0d902c8cc032308262fa3a3befdc0687",
}
PYTHON_PROTOCOL_FILES = [
    "run_manifest.py", "cptac_eval.py", "hest_eval.py", "pathorob_eval.py", "thunder_eval.py",
]
PATCH_FILES = ["thunder.patch", "hest.patch", "patho_bench.patch", "pathorob.patch"]
TUNING_SETTINGS = {"EXTRACTION_BATCH", "ATTACK_BATCH"}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_model_settings(model_path=REPO_ROOT / "model.py"):
    tree = ast.parse(Path(model_path).read_text())
    names = {
        "MODEL_NAME", "AMP_DTYPE", "EXTRACTION_BATCH", "ATTACK_BATCH",
        "CHECKPOINT", "MODEL_REVISION", "MODEL_ASSETS",
    }
    settings = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in names
    }
    assert isinstance(settings["MODEL_NAME"], str)
    assert settings["MODEL_NAME"] == os.path.basename(settings["MODEL_NAME"])
    assert settings["MODEL_NAME"] not in ("", ".", "..")
    assert settings["AMP_DTYPE"] in ("float16", "bfloat16", "float32")
    assert settings["EXTRACTION_BATCH"] > 0 and settings["ATTACK_BATCH"] > 0
    return settings


def build_manifest(model_path=REPO_ROOT / "model.py", full_asset_hashes=True):
    model_path = Path(model_path)
    settings = read_model_settings(model_path)
    tree = ast.parse(model_path.read_text())
    tree.body = [
        node for node in tree.body
        if not (
            isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in TUNING_SETTINGS
        )
    ]
    adapter_sha256 = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()

    asset_paths = []
    if "CHECKPOINT" in settings:
        asset_paths.append(Path(settings["CHECKPOINT"]))
        checkpoint_source = Path(settings["CHECKPOINT"]).parent / "labless_source/model.py"
        if checkpoint_source.exists():
            asset_paths.append(checkpoint_source)
    asset_paths.extend(Path(path) for path in settings.get("MODEL_ASSETS", []))
    assert asset_paths or settings.get("MODEL_REVISION")
    assert len(set(asset_paths)) == len(asset_paths)
    assets = []
    for path in asset_paths:
        assert path.is_absolute() and path.is_file(), path
        asset = {"path": str(path), "bytes": path.stat().st_size}
        if full_asset_hashes:
            asset["sha256"] = _sha256(path)
        assets.append(asset)

    python_hashes = {}
    for name in PYTHON_PROTOCOL_FILES:
        protocol_tree = ast.parse((REPO_ROOT / name).read_text())
        python_hashes[name] = hashlib.sha256(
            ast.dump(protocol_tree, include_attributes=False).encode()
        ).hexdigest()
    patch_hashes = {name: _sha256(REPO_ROOT / name) for name in PATCH_FILES}
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model_name": settings["MODEL_NAME"],
        "model_revision": settings.get("MODEL_REVISION"),
        "amp_dtype": settings["AMP_DTYPE"],
        "adapter_sha256": adapter_sha256,
        "assets": assets,
        "benchmark_commits": BENCHMARKS,
        "python_protocol_sha256": python_hashes,
        "patch_sha256": patch_hashes,
    }


def _existing_outputs(run_root, model_name):
    model_id = f"{model_name}_optimized"
    candidates = [
        run_root / "hest/results" / model_name,
        run_root / "hest/embeddings" / model_name,
        run_root / "cptac" / model_name,
        run_root / "pathorob/features" / f"{model_name}_clsmean",
        run_root / "pathorob/results/robustness_index" / f"{model_name}_clsmean",
        run_root / "pathorob/results/apd" / f"{model_name}_clsmean",
        run_root / "pathorob/results/clustering_score" / f"{model_name}_clsmean",
    ]
    candidates.extend(run_root.glob(f"thunder/embeddings/*/{model_id}"))
    candidates.extend(run_root.glob(f"thunder/outputs/res/*/{model_id}"))
    return [str(path) for path in candidates if path.exists()]


def write_or_verify_manifest(model_path=REPO_ROOT / "model.py", run_root=RUN_ROOT):
    manifest = build_manifest(model_path, full_asset_hashes=True)
    run_root = Path(run_root)
    manifest_dir = run_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    target = manifest_dir / f'{manifest["model_name"]}.json'
    with (manifest_dir / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            saved = json.loads(target.read_text())
            assert saved == manifest, (
                f"MODEL_NAME={manifest['model_name']!r} is already bound to a different model or "
                "evaluation protocol; choose a new MODEL_NAME"
            )
        else:
            existing = _existing_outputs(run_root, manifest["model_name"])
            assert not existing, f"unverifiable outputs exist without a manifest: {existing}"
            temporary = target.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, target)
    return target


def verify_manifest(model_path=REPO_ROOT / "model.py", run_root=RUN_ROOT):
    current = build_manifest(model_path, full_asset_hashes=False)
    target = Path(run_root) / "manifests" / f'{current["model_name"]}.json'
    assert target.exists(), "run preflight before evaluation stages"
    saved = json.loads(target.read_text())
    current_assets = current.pop("assets")
    saved_assets = saved.pop("assets")
    assert current == saved, (
        f"MODEL_NAME={current['model_name']!r} no longer matches its run manifest; "
        "choose a new MODEL_NAME"
    )
    assert [
        {"path": asset["path"], "bytes": asset["bytes"]} for asset in saved_assets
    ] == current_assets
    return target


if __name__ == "__main__":
    assert len(sys.argv) == 2 and sys.argv[1] in ("write", "verify")
    path = write_or_verify_manifest() if sys.argv[1] == "write" else verify_manifest()
    print(f"PASS run manifest {path}", flush=True)
