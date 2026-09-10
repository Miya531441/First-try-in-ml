"""Arena geometry: outer walls plus mirrored obstacles of two classes, batched over envs.

Class 0 = full wall (blocks vision, shots and movement); class 1 = low crate
(blocks movement only)."""
from __future__ import annotations

import numpy as np

WALL, CRATE = 0, 1


class Arena:
    def __init__(self, num_envs: int, size: float, max_obstacles: int):
        self.num_envs = num_envs
        self.size = float(size)
        self.max_obstacles = int(max_obstacles)
        self.boxes = np.zeros((num_envs, max_obstacles, 4), dtype=np.float32)
        self.kind = np.zeros((num_envs, max_obstacles), dtype=np.int64)
        self.mask = np.zeros((num_envs, max_obstacles), dtype=bool)

    def set(self, env_idx, boxes_list, kinds_list):
        for e, boxes, kinds in zip(env_idx, boxes_list, kinds_list):
            m = min(len(boxes), self.max_obstacles)
            self.boxes[e] = 0.0
            self.mask[e] = False
            self.kind[e] = 0
            if m:
                self.boxes[e, :m] = boxes[:m]
                self.kind[e, :m] = kinds[:m]
                self.mask[e, :m] = True

    @property
    def wall_mask(self) -> np.ndarray:
        """Boxes that occlude vision and stop shots."""
        return self.mask & (self.kind == WALL)

    def mirror_boxes(self, boxes: np.ndarray) -> np.ndarray:
        out = np.empty_like(boxes)
        out[..., 0] = self.size - boxes[..., 2]
        out[..., 1] = self.size - boxes[..., 3]
        out[..., 2] = self.size - boxes[..., 0]
        out[..., 3] = self.size - boxes[..., 1]
        return out

    def encoding(self, mirrored: bool = False) -> np.ndarray:
        """[E, M*6]: (cx, cy, w, h, is_wall, present) per slot, normalised."""
        b = self.mirror_boxes(self.boxes) if mirrored else self.boxes
        cx = (b[..., 0] + b[..., 2]) / 2.0 / self.size
        cy = (b[..., 1] + b[..., 3]) / 2.0 / self.size
        w = (b[..., 2] - b[..., 0]) / self.size
        h = (b[..., 3] - b[..., 1]) / self.size
        enc = np.stack([cx, cy, w, h, (self.kind == WALL).astype(np.float32), self.mask.astype(np.float32)], -1)
        enc = enc * self.mask[..., None]
        return enc.reshape(self.num_envs, -1).astype(np.float32)

    @staticmethod
    def point_in_boxes(points, boxes, mask, margin: float = 0.0):
        p = points[:, :, None, :]
        lo = boxes[:, None, :, 0:2] - margin
        hi = boxes[:, None, :, 2:4] + margin
        inside = (p >= lo).all(-1) & (p <= hi).all(-1) & mask[:, None, :]
        return inside.any(-1)
