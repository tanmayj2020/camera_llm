"""Person Re-Identification embedding extractor.

Priority chain:  CLIP ViT-B/16 → DINOv2 ViT-B/14 → ResNet50.

• CLIP ViT-B/16  — 512-dim, CLIP-ReID (CVPR 2023), best zero-shot transfer.
• DINOv2 ViT-B/14 — 768-dim (projected to 512), Meta self-supervised,
  SOTA visual features for person ReID on Market-1501 / MSMT17 (2024-2026).
• ResNet50        — 512-dim, classic fallback when neither is available.
"""

import base64
import io
import logging

import numpy as np

logger = logging.getLogger(__name__)


class ReIDEmbedder:
    """Extracts Re-ID embeddings from person crops.

    Priority: CLIP ViT-B/16 → DINOv2 ViT-B/14 → ResNet50.
    """

    EMBED_DIM = 512

    def __init__(self):
        self._model = None
        self._preprocess = None
        self._available = None
        self._backend = None

    def _load(self) -> bool:
        if self._available is not None:
            return self._available

        # Try CLIP ViT-B/16 first (SOTA for ReID)
        try:
            import torch
            import clip as clip_module

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip_module.load("ViT-B/16", device=device)
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._device = device
            self._backend = "clip-vit-b16"
            self._available = True
            logger.info("ReID embedder loaded (CLIP ViT-B/16 → %d-dim on %s)", self.EMBED_DIM, device)
            return True
        except Exception as e:
            logger.debug("CLIP unavailable (%s), trying DINOv2 fallback", e)

        # Fallback 1: DINOv2 ViT-B/14 (Meta, self-supervised — SOTA visual features 2024-2026)
        try:
            import torch
            import torchvision.transforms as T

            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            # Project DINOv2's 768-dim CLS to 512-dim for compatibility
            proj = torch.nn.Linear(768, self.EMBED_DIM).to(device)
            torch.nn.init.orthogonal_(proj.weight)

            self._model = model
            self._proj = proj
            self._preprocess = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._device = device
            self._backend = "dinov2-vitb14"
            self._available = True
            logger.info("ReID embedder loaded (DINOv2 ViT-B/14 → %d-dim on %s)", self.EMBED_DIM, device)
            return True
        except Exception as e:
            logger.debug("DINOv2 unavailable (%s), trying ResNet50 fallback", e)

        # Fallback 2: ResNet50
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as T

            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            backbone.fc = torch.nn.Linear(backbone.fc.in_features, self.EMBED_DIM)
            backbone.eval()
            self._model = backbone
            self._preprocess = T.Compose([
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._device = "cpu"
            self._backend = "resnet50"
            self._available = True
            logger.info("ReID embedder loaded (ResNet50 fallback → %d-dim)", self.EMBED_DIM)
            return True
        except Exception as e:
            logger.warning("ReID embedder unavailable: %s", e)
            self._available = False
        return self._available

    def extract(self, keyframe_b64: str, bbox: list[float]) -> np.ndarray | None:
        """Extract embedding from a person crop within a keyframe.

        Args:
            keyframe_b64: base64-encoded JPEG keyframe
            bbox: [x1, y1, x2, y2] bounding box
        Returns:
            512-dim L2-normalized embedding or None
        """
        if not self._load():
            return None
        try:
            import torch
            from PIL import Image

            img_bytes = base64.b64decode(keyframe_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            x1, y1, x2, y2 = [int(c) for c in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 - x1 < 10 or y2 - y1 < 10:
                return None
            crop = img.crop((x1, y1, x2, y2))

            if self._backend == "clip-vit-b16":
                tensor = self._preprocess(crop).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    emb = self._model.encode_image(tensor).squeeze(0).float()
                    emb = emb / (emb.norm() + 1e-8)
                return emb.cpu().numpy()
            elif self._backend == "dinov2-vitb14":
                tensor = self._preprocess(crop).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    features = self._model(tensor)  # (1, 768)
                    emb = self._proj(features).squeeze(0)
                    emb = emb / (emb.norm() + 1e-8)
                return emb.cpu().numpy()
            else:
                tensor = self._preprocess(crop).unsqueeze(0)
                with torch.no_grad():
                    emb = self._model(tensor).squeeze(0)
                    emb = emb / (emb.norm() + 1e-8)
                return emb.numpy()
        except Exception as e:
            logger.debug("ReID extraction failed: %s", e)
            return None
