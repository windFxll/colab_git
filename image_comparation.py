from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re


project_root = Path(__file__).resolve().parent

gt_dir = project_root / "test_pattern" / "simulation_results" / "figure"

DRIVE_ROOT = Path("/content/drive/MyDrive/Colab Notebooks")

exp_name = "exp_unet_test_bottleneck_bce_dice_0.5mse"
exp_dir = DRIVE_ROOT / "experiments" / exp_name


def load_binary(path, thresh=127):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)

    return (img < thresh).astype(np.uint8)


def pixel_difference(pred, target):
    diff = np.abs(pred.astype(np.int32) - target.astype(np.int32))
    return diff.mean()


def row_max_gap(binary):
    h, w = binary.shape
    gaps = np.zeros(h)

    for y in range(h):
        row = binary[y]

        max_len = 0
        cur = 0

        for v in row:
            if v == 1:
                cur += 1
                max_len = max(max_len, cur)
            else:
                cur = 0

        gaps[y] = max_len

    return gaps


def min_gap(binary):
    gaps = row_max_gap(binary)
    valid = gaps[gaps > 0]

    if len(valid) == 0:
        return 0

    return valid.min()


def gap_profile_error(pred, target):
    gp = row_max_gap(pred)
    gt = row_max_gap(target)

    valid = (gt > 0)

    if valid.sum() == 0:
        return 0.0

    return np.mean(np.abs(gp[valid] - gt[valid]))


def evaluate_pair(pred_path, gt_path):
    pred = load_binary(pred_path)
    gt = load_binary(gt_path)

    px_diff = pixel_difference(pred, gt)

    pred_gap = min_gap(pred)
    gt_gap = min_gap(gt)
    gap_err = abs(pred_gap - gt_gap)

    profile_err = gap_profile_error(pred, gt)

    return {
        "pixel_diff": px_diff,
        "gap_err": gap_err,
        "profile_err": profile_err,
    }

def extract_epoch(folder_name):
    m = re.search(r"epoch_(\d+)", folder_name)
    if m:
        return int(m.group(1))
    return -1


def main():
    infer_dirs = sorted(
        [p for p in exp_dir.iterdir() if p.is_dir() and "infer_checkpoint" in p.name],
        key=lambda x: extract_epoch(x.name),
    )

    epochs = []
    pixel_curve = []
    gap_curve = []
    profile_curve = []

    for infer_dir in infer_dirs:
        pixel_list = []
        gap_list = []
        profile_list = []

        for i in range(10):
            gt_path = gt_dir / f"resist_bottom_{i:05d}.png"
            pred_path = infer_dir / f"test_pattern_{i:05d}.png"

            if not gt_path.exists():
                print(f"[WARN] missing gt: {gt_path}")
                continue

            if not pred_path.exists():
                print(f"[WARN] missing pred: {pred_path}")
                continue

            result = evaluate_pair(pred_path, gt_path)

            pixel_list.append(result["pixel_diff"])
            gap_list.append(result["gap_err"])
            profile_list.append(result["profile_err"])

        if len(pixel_list) == 0:
            continue

        epoch = extract_epoch(infer_dir.name)

        epochs.append(epoch)
        pixel_curve.append(np.mean(pixel_list))
        gap_curve.append(np.mean(gap_list))
        profile_curve.append(np.mean(profile_list))

        print(
            f"[Epoch {epoch}] "
            f"pixel={pixel_curve[-1]:.4f}, "
            f"gap={gap_curve[-1]:.4f}, "
            f"profile={profile_curve[-1]:.4f}"
        )

    # ==================================================
    # 绘图
    # ==================================================
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # 左轴：gap 和 profile
    ax1.plot(epochs, gap_curve, marker="o", label="Min Gap Error")
    ax1.plot(epochs, profile_curve, marker="o", label="Gap Profile Error")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Gap / Profile Error")
    ax1.grid(True)

    # 右轴：pixel
    ax2 = ax1.twinx()
    ax2.plot(
        epochs,
        pixel_curve,
        marker="o",
        linestyle="--",
        color="red",
        label="Pixel Difference",
    )
    ax2.set_ylabel("Pixel Difference", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("Prediction Quality vs Training Epoch")

    save_path = exp_dir / "evaluation_trend.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved figure to: {save_path}")


if __name__ == "__main__":
    main()