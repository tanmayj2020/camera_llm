"""Person Re-Identification embedding extractor using ResNet50 backbone.

Extracts 128-dim appearance embeddings from cropped person bounding boxes
for cross-camera identity matching.
"""

import base64
import io
import logging

import numpy as np

logger = logging.getLogger(__name__)


class ReIDEmbedder:
    """Extracts Re-ID embeddings from person crops. Lazy-loads ResNet50 backbone."""

    EMBED_DIM = 128

    def __init__(self):
        self._model = None
        self._transform = None
        self._available = None

    def _load(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as T

            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # Replace classifier with 128-dim projection
            backbone.fc = torch.nn.Linear(backbone.fc.in_features, self.EMBED_DIM)
            backbone.eval()
            self._model = backbone
            self._transform = T.Compose([
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._available = True
            logger.info("ReID embedder loaded (ResNet50 → %d-dim)", self.EMBED_DIM)
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
            128-dim L2-normalized embedding or None
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
            tensor = self._transform(crop).unsqueeze(0)

            with torch.no_grad():
                emb = self._model(tensor).squeeze(0)
                emb = emb / (emb.norm() + 1e-8)
            return emb.numpy()
        except Exception as e:
            logger.debug("ReID extraction failed: %s", e)
            return None
