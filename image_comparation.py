from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re


project_root = Path(__file__).resolve().parent

gt_dir = project_root / "test_pattern" / "simulation_results" / "figure"

DRIVE_ROOT = Path("/content/drive/MyDrive/Colab Notebooks")

exp_name = "exp_unet_edge_dilated_64_bce_dice_0.5mse"
exp_dir = DRIVE_ROOT / "experiments" / exp_name


def load_binary(path, thresh=127):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)

    return (img < thresh).astype(np.uint8)


def mse_difference(pred, target):
    diff = pred.astype(np.float32) - target.astype(np.float32)
    return np.mean(diff ** 2)


def row_min_bridge(binary):
    h, w = binary.shape
    bridges = np.full(h, np.nan)

    for y in range(h):
        row = binary[y]

        black_runs = []
        in_black = False
        start = 0

        for x in range(w):
            if row[x] == 1 and not in_black:
                start = x
                in_black = True
            elif row[x] == 0 and in_black:
                black_runs.append((start, x - 1))
                in_black = False

        if in_black:
            black_runs.append((start, w - 1))

        if len(black_runs) < 2:
            continue

        local_bridges = []

        for i in range(len(black_runs) - 1):
            right_end = black_runs[i][1]
            next_left = black_runs[i + 1][0]

            bridge = next_left - right_end - 1

            if bridge > 0:
                local_bridges.append(bridge)

        if len(local_bridges) > 0:
            bridges[y] = np.mean(local_bridges)

    return bridges


def mean_gap(binary):
    bridges = row_min_bridge(binary)
    valid = bridges[~np.isnan(bridges)]

    if len(valid) == 0:
        return 0.0

    return valid.mean()


def gap_profile_error(pred, target):
    gp = row_min_bridge(pred)
    gt = row_min_bridge(target)

    valid = (~np.isnan(gp)) & (~np.isnan(gt))

    if valid.sum() == 0:
        return 0.0

    return np.mean(np.abs(gp[valid] - gt[valid]))


def evaluate_pair(pred_path, gt_path):
    pred = load_binary(pred_path)
    gt = load_binary(gt_path)

    mse_diff = mse_difference(pred, gt)

    pred_gap = mean_gap(pred)
    gt_gap = mean_gap(gt)
    gap_err = abs(pred_gap - gt_gap)

    profile_err = gap_profile_error(pred, gt)

    return {
        "mse_diff": mse_diff,
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
    mse_curve = []
    gap_curve = []
    profile_curve = []

    for infer_dir in infer_dirs:
        mse_list = []
        gap_list = []
        profile_list = []

        for i in range(10):
            gt_path = gt_dir / f"resist_bottom_{i:05d}.png"
            pred_path = infer_dir / f"test_pattern_{i:05d}.png"
            # pred_path = infer_dir / f"resist_bottom_{i:05d}.png"

            if not gt_path.exists():
                print(f"[WARN] missing gt: {gt_path}")
                continue

            if not pred_path.exists():
                print(f"[WARN] missing pred: {pred_path}")
                continue

            result = evaluate_pair(pred_path, gt_path)

            mse_list.append(result["mse_diff"])
            gap_list.append(result["gap_err"])
            profile_list.append(result["profile_err"])

        if len(mse_list) == 0:
            continue

        epoch = extract_epoch(infer_dir.name)

        epochs.append(epoch)
        mse_curve.append(np.mean(mse_list))
        gap_curve.append(np.mean(gap_list))
        profile_curve.append(np.mean(profile_list))

        print(
            f"[Epoch {epoch}] "
            f"mse={mse_curve[-1]:.4f}, "
            f"gap={gap_curve[-1]:.4f}, "
            f"profile={profile_curve[-1]:.4f}"
        )

    # ==================================================
    # 绘图
    # ==================================================
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # 左轴：gap 和 profile
    ax1.plot(epochs, gap_curve, marker="o", label="Mean Gap Error")
    ax1.plot(epochs, profile_curve, marker="o", label="Gap Profile Error")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Gap / Profile Error")
    ax1.grid(True)

    # 右轴：MSE
    ax2 = ax1.twinx()

    ax2.plot(
        epochs,
        mse_curve,
        marker="s",
        linestyle="--",
        color="green",
        label="MSE",
    )

    ax2.set_ylabel("MSE")
    ax2.tick_params(axis="y", labelcolor="green")

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