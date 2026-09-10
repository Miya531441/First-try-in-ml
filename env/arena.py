"""Arena geometry: outer walls plus mirrored axis-aligned obstacles, batched over envs."""
from __future__ import annotations

import numpy as np


class Arena:
    """Holds per-env obstacle boxes as padded arrays [E, M, 4] with a mask [E, M].

    Boxes are (xmin, ymin, xmax, ymax).  Obstacles are point-mirrored about the arena
    centre so that both teams face an identical layout (see ``mirror_boxes``).
    """

    def __init__(self, num_envs: int, size: float, max_obstacles: int):
        self.num_envs = num_envs
        self.size = float(size)
        self.max_obstacles = int(max_obstacles)
        self.boxes = np.zeros((num_envs, max_obstacles, 4), dtype=np.float32)
        self.mask = np.zeros((num_envs, max_obstacles), dtype=bool)

    def set(self, env_idx: np.ndarray, boxes_list):
        for e, boxes in zip(env_idx, boxes_list):
            m = len(boxes)
            self.boxes[e] = 0.0
            self.mask[e] = False
            if m:
                self.boxes[e, :m] = boxes
                self.mask[e, :m] = True

    def mirror_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Point-reflect boxes about the centre: (x, y) -> (S - x, S - y)."""
        out = np.empty_like(boxes)
        out[..., 0] = self.size - boxes[..., 2]
        out[..., 1] = self.size - boxes[..., 3]
        out[..., 2] = self.size - boxes[..., 0]
        out[..., 3] = self.size - boxes[..., 1]
        return out

    def encoding(self, mirrored: bool = False) -> np.ndarray:
        """Normalised obstacle encoding for the centralised critic: [E, M*5]
        (cx, cy, w, h, present), optionally in the mirrored (team B) frame."""
        b = self.mirror_boxes(self.boxes) if mirrored else self.boxes
        cx = (b[..., 0] + b[..., 2]) / 2.0 / self.size
        cy = (b[..., 1] + b[..., 3]) / 2.0 / self.size
        w = (b[..., 2] - b[..., 0]) / self.size
        h = (b[..., 3] - b[..., 1]) / self.size
        enc = np.stack([cx, cy, w, h, self.mask.astype(np.float32)], -1) * self.mask[..., None]
        return enc.reshape(self.num_envs, -1).astype(np.float32)

    @staticmethod
    def point_in_boxes(points: np.ndarray, boxes: np.ndarray, mask: np.ndarray, margin: float = 0.0):
        """points [E,P,2], boxes [E,M,4] -> inside [E,P] (any box, inflated by margin)."""
        p = points[:, :, None, :]
        lo = boxes[:, None, :, 0:2] - margin
        hi = boxes[:, None, :, 2:4] + margin
        inside = (p >= lo).all(-1) & (p <= hi).all(-1) & mask[:, None, :]
        return inside.any(-1)
