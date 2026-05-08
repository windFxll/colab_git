import os
import glob
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def extract_tip_candidates(binary_img, border_margin=2):
    """
    binary_img: uint8, foreground=255
    return: [(x, y), ...]
    """
    h, w = binary_img.shape

    contours, _ = cv2.findContours(
        binary_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    tips = []

    for cnt in contours:
        if len(cnt) < 5:
            continue

        pts = cnt[:, 0, :]
        xs = pts[:, 0]

        xmin = xs.min()
        xmax = xs.max()

        # left short edge center
        if xmin > border_margin:
            left_pts = pts[np.abs(xs - xmin) <= 1]
            left_y = int(np.mean(left_pts[:, 1]))
            tips.append((int(xmin), left_y))

        # right short edge center
        if xmax < w - 1 - border_margin:
            right_pts = pts[np.abs(xs - xmax) <= 1]
            right_y = int(np.mean(right_pts[:, 1]))
            tips.append((int(xmax), right_y))

    return tips


def pair_tip_candidates(tips, y_thresh=20):
    """
    同一行 tip 配对
    """
    if len(tips) < 2:
        return []

    tips = sorted(tips, key=lambda p: p[1])

    used = set()
    pairs = []

    for i, p1 in enumerate(tips):
        if i in used:
            continue

        best_j = None
        best_dx = 1e9

        for j, p2 in enumerate(tips):
            if j <= i or j in used:
                continue

            if abs(p1[1] - p2[1]) > y_thresh:
                continue

            dx = abs(p2[0] - p1[0])

            if dx < best_dx:
                best_dx = dx
                best_j = j

        if best_j is not None:
            used.add(i)
            used.add(best_j)
            pairs.append((p1, tips[best_j]))

    return pairs


def build_tip_weight_map(
    binary_img,
    roi_height=80,
    pad_x=15,
    tip_weight=3.0,
):
    """
    根据 layout 生成 tip 权重图
    """
    h, w = binary_img.shape

    weight = np.ones((h, w), dtype=np.float32)

    tips = extract_tip_candidates(binary_img)
    pairs = pair_tip_candidates(tips)

    for p1, p2 in pairs:
        x1, y1 = p1
        x2, y2 = p2

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        gap = abs(x2 - x1)

        roi_w = gap + 2 * pad_x
        roi_h = roi_height

        x0 = max(0, cx - roi_w // 2)
        x1 = min(w, cx + roi_w // 2)

        y0 = max(0, cy - roi_h // 2)
        y1 = min(h, cy + roi_h // 2)

        weight[y0:y1, x0:x1] = tip_weight

    return weight


class LithoDataset(Dataset):
    def __init__(self, root_dir, pattern_types):
        self.root_dir = os.path.abspath(root_dir)
        self.pattern_types = pattern_types
        self.pairs = []

        for pattern in self.pattern_types:
            layout_dir = self._layout_dir(pattern)
            resist_dir = self._resist_dir(pattern)

            self._ensure_png_dir(layout_dir, "layout", pattern)
            self._ensure_png_dir(resist_dir, "resist_bottom", pattern)

            input_paths = sorted(glob.glob(os.path.join(layout_dir, "*.png")))

            print(f"[DEBUG] {pattern}: input={len(input_paths)}")

            if not input_paths:
                raise RuntimeError(
                    f"No layout samples found for pattern '{pattern}' in '{layout_dir}'."
                )

            for p in input_paths:
                filename = os.path.basename(p)

                self._extract_sequence_id(filename)

                tgt = os.path.join(resist_dir, filename)

                if not os.path.exists(tgt):
                    raise FileNotFoundError(
                        f"Missing paired resist image for pattern '{pattern}', "
                        f"file '{filename}': '{tgt}'"
                    )

                self._ensure_readable_image(p, "layout")
                self._ensure_readable_image(tgt, "resist_bottom")

                self.pairs.append((p, tgt))

        if len(self.pairs) == 0:
            raise RuntimeError(
                "No paired samples found. "
                "Please check layout/resist directories and file names."
            )

        # debug 输出前几张
        self.debug_saved = 0

    def _layout_dir(self, pattern):
        return os.path.join(self.root_dir, "layout", pattern, "png")

    def _resist_dir(self, pattern):
        preferred_dir = os.path.join(
            self.root_dir,
            "output",
            pattern,
            "resist_bottom",
            "png",
        )

        legacy_dir = os.path.join(
            self.root_dir,
            "output",
            pattern,
            "resist_contour",
            "figure",
        )

        if os.path.isdir(preferred_dir):
            return preferred_dir

        if os.path.isdir(legacy_dir):
            return legacy_dir

        return preferred_dir

    def _ensure_png_dir(self, path, data_kind, pattern):
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Expected {data_kind} directory for pattern '{pattern}' "
                f"does not exist: '{path}'"
            )

        if not glob.glob(os.path.join(path, "*.png")):
            raise RuntimeError(
                f"No png files found in {data_kind} directory "
                f"for pattern '{pattern}': '{path}'"
            )

    def _ensure_readable_image(self, path, data_kind):
        img = cv2.imread(path, 0)

        if img is None:
            raise RuntimeError(
                f"Unreadable {data_kind} image file: '{path}'"
            )

    def _extract_sequence_id(self, filename):
        match = re.search(r"(\d{5})", filename)

        if match is None:
            raise ValueError(
                f"Cannot extract 5-digit sequence id from: {filename}"
            )

        return match.group(1)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp_path, tgt_path = self.pairs[idx]

        inp = cv2.imread(inp_path, 0)
        tgt = cv2.imread(tgt_path, 0)

        if inp is None or tgt is None:
            raise RuntimeError(
                f"Failed to read image pair: {inp_path}, {tgt_path}"
            )

        # black polygon -> foreground
        binary = (inp < 127).astype(np.uint8) * 255

        weight = build_tip_weight_map(
            binary,
            roi_height=80,
            pad_x=25,
            tip_weight=3.0,
        )

        # debug overlay
        # if self.debug_saved < 20:
        #     print(f"DEBUG SAVE at idx = {idx}")

        #     vis = cv2.cvtColor(inp, cv2.COLOR_GRAY2BGR)

        #     roi_mask = weight > 1.0

        #     overlay = vis.copy()
        #     overlay[roi_mask] = (220, 220, 220)

        #     alpha = 0.45

        #     vis = cv2.addWeighted(
        #         overlay,
        #         alpha,
        #         vis,
        #         1 - alpha,
        #         0,
        #     )

        #     tips = extract_tip_candidates(binary)

        #     for x, y in tips:
        #         cv2.circle(
        #             vis,
        #             (x, y),
        #             3,
        #             (0, 0, 255),
        #             -1,
        #         )

        #     cv2.imwrite(
        #         f"debug_overlay_{self.debug_saved}_idx{idx}.png",
        #         vis,
        #     )

        #     self.debug_saved += 1

        inp = inp.astype(np.float32) / 255.0
        tgt = tgt.astype(np.float32) / 255.0

        inp = torch.tensor(inp).unsqueeze(0)
        tgt = torch.tensor(tgt).unsqueeze(0)
        weight = torch.tensor(weight).unsqueeze(0)

        return inp, tgt, weight