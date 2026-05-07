import os
import glob
import re
import cv2
import torch
from torch.utils.data import Dataset

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
              self._extract_sequence_id(filename)  # 仅做编号合法性检查

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

    def _layout_dir(self, pattern):
        return os.path.join(self.root_dir, "layout", pattern, "png")

    def _resist_dir(self, pattern):
        preferred_dir = os.path.join(
            self.root_dir, "output", pattern, "resist_bottom", "png"
        )
        legacy_dir = os.path.join(
            self.root_dir, "output", pattern, "resist_contour", "figure"
        )
        if os.path.isdir(preferred_dir):
            return preferred_dir
        if os.path.isdir(legacy_dir):
            return legacy_dir
        return preferred_dir

    def _ensure_png_dir(self, path, data_kind, pattern):
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Expected {data_kind} directory for pattern '{pattern}' does not exist: '{path}'"
            )
        if not glob.glob(os.path.join(path, "*.png")):
            raise RuntimeError(
                f"No png files found in {data_kind} directory for pattern '{pattern}': '{path}'"
            )

    def _ensure_readable_image(self, path, data_kind):
        img = cv2.imread(path, 0)
        if img is None:
            raise RuntimeError(f"Unreadable {data_kind} image file: '{path}'")

    def _extract_sequence_id(self, filename):
        match = re.search(r"(\d{5})", filename)
        if match is None:
            raise ValueError(f"Cannot extract 5-digit sequence id from: {filename}")
        return match.group(1)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp_path, tgt_path = self.pairs[idx]

        inp = cv2.imread(inp_path, 0)
        tgt = cv2.imread(tgt_path, 0)
        if inp is None or tgt is None:
            raise RuntimeError(f"Failed to read image pair: {inp_path}, {tgt_path}")
        inp = inp / 255.0
        tgt = tgt / 255.0

        inp = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
        tgt = torch.tensor(tgt, dtype=torch.float32).unsqueeze(0)

        return inp, tgt