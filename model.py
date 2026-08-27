import importlib.util
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

CHECKPOINT = "/data/USER/nanopath/main/RUN/latest.pt"
MODEL_NAME = "replace-with-unique-model-name"
EXTRACTION_BATCH = 2048
ATTACK_BATCH = 512
AMP_DTYPE = "float16"


class EvalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        source_path = Path(CHECKPOINT).parent / "labless_source/model.py"
        spec = importlib.util.spec_from_file_location("nanopath_checkpoint_model", source_path)
        assert spec is not None and spec.loader is not None
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)

        checkpoint = torch.load(
            CHECKPOINT, map_location="cpu", weights_only=True, mmap=True,
        )
        config = checkpoint["config"]
        self.backbone = source.DinoV2ViT(
            variant=config["model"]["type"],
            drop_path_rate=config["dino"]["drop_path_rate"],
        )
        self.backbone.load_state_dict(checkpoint["model_ema"], strict=True)
        mean = tuple(config["data"]["mean"])
        std = tuple(config["data"]["std"])
        assert len(mean) == len(std) == 3 and all(value > 0 for value in std)
        del checkpoint
        self.backbone.eval().cuda()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.name = MODEL_NAME
        assert self.name == os.path.basename(self.name) and self.name not in (".", "..")
        base_transform = source.probe_transforms()[0]
        normalize = transforms.Normalize(mean=mean, std=std)
        self._transform = transforms.Compose([base_transform, normalize])
        self._to_tensor = transforms.Compose([transforms.ToTensor(), normalize])

        # Nanopath's native probe applies checkpoint-configured normalization after
        # probe_transforms(). Keep that split contract explicit and fail preflight if
        # the saved source starts normalizing internally.
        test_image = Image.new("RGB", (224, 224), (32, 128, 224))
        raw = base_transform(test_image)
        assert raw.shape == (3, 224, 224) and raw.min() >= 0 and raw.max() <= 1
        expected = (raw - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
        assert torch.equal(self._transform(test_image), expected)
        assert torch.equal(self._to_tensor(test_image), expected)

        # The checkpoint's saved source is authoritative: some Nanopath variants fuse
        # several block outputs and/or return a denser segmentation grid. Discover that
        # interface once instead of duplicating model-specific rules in this repository.
        with torch.inference_mode(), torch.autocast(
            "cuda", getattr(torch, AMP_DTYPE), enabled=AMP_DTYPE != "float32",
        ):
            image = torch.zeros(1, 3, 224, 224, device="cuda")
            classification = self.classification_features(image)
            segmentation = self.segmentation_features(image)
        assert classification.ndim == 2 and classification.shape[0] == 1
        assert segmentation.ndim == 3 and segmentation.shape[0] == 1
        self.classification_dim = classification.shape[-1]
        self.segmentation_tokens = segmentation.shape[1]
        self.segmentation_dim = segmentation.shape[-1]
        self.clsmean_dim = 2 * self.backbone.embed_dim
        del image, classification, segmentation

    def classification_features(self, images):
        return self.backbone.probe_features(images)

    def segmentation_features(self, images):
        return self.backbone.encode_image(images)[:, self.backbone.registers:]

    def clsmean_features(self, images):
        features = self.backbone(images)
        return torch.cat([
            features["x_norm_clstoken"], features["x_norm_patchtokens"].mean(1),
        ], dim=-1)

    def forward(self, images):
        return self.classification_features(images)

    def transform(self, resize=True, timm_style=False):
        return self._transform if resize else self._to_tensor


if __name__ == "__main__":
    model = EvalModel()
    image = Image.new("RGB", (224, 224), (32, 128, 224))
    resized = model.transform(timm_style=True)(image)
    unresized = model.transform(resize=False)(image)
    assert torch.equal(resized, unresized)
    assert resized.min() < 0 and resized.max() > 0
    images = resized.unsqueeze(0).repeat(2, 1, 1, 1).cuda()
    with torch.inference_mode(), torch.autocast(
        "cuda", getattr(torch, AMP_DTYPE), enabled=AMP_DTYPE != "float32",
    ):
        classification = model.classification_features(images)
        segmentation = model.segmentation_features(images)
        clsmean = model.clsmean_features(images)
    assert model.name == MODEL_NAME
    assert classification.shape == (2, model.classification_dim)
    assert segmentation.shape == (2, model.segmentation_tokens, model.segmentation_dim)
    assert clsmean.shape == (2, model.clsmean_dim)
    assert torch.isfinite(classification).all()
    assert torch.isfinite(segmentation).all()
    assert torch.isfinite(clsmean).all()
    images.requires_grad = True
    with torch.autocast(
        "cuda", getattr(torch, AMP_DTYPE), enabled=AMP_DTYPE != "float32",
    ):
        model.classification_features(images).sum().backward()
    assert images.grad is not None and torch.isfinite(images.grad).all()
    print(
        f"PASS {MODEL_NAME}: classification={tuple(classification.shape)} "
        f"segmentation={tuple(segmentation.shape)} clsmean={tuple(clsmean.shape)} "
        f"dtype={AMP_DTYPE} normalized_input=[{resized.min():.3f}, {resized.max():.3f}]",
        flush=True,
    )
