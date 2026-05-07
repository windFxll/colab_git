import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# ===== 项目根目录 =====
project_root = Path(__file__).resolve().parent


def load_image(path):
    img = Image.open(path).convert('L')
    img = np.array(img).astype(np.float32)

    if img.max() > 1.0:
        img = img / 255.0

    return img


def compute_metrics(gt, pred):
    diff = pred - gt

    l2 = np.sqrt(np.sum(diff ** 2))
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)

    return l2, mse, rmse

def compute_soft_connectivity(gt, pred, threshold=0.5, d_max=5):
    """
    软连接指标：
    衡量预测在“潜在连接区域”中的响应强度
    """
    gt_bin = (gt > threshold).astype(np.uint8)

    from scipy.ndimage import distance_transform_edt
    dist_map = distance_transform_edt(1 - gt_bin)

    # 只关注“接近结构但不是结构本身”的区域
    bridge_region = (dist_map < d_max) & (dist_map > 0)

    if np.sum(bridge_region) == 0:
        return 0.0

    return float(np.mean((pred > 0.3)[bridge_region]))

def compute_soft_connectivity_v2(gt, pred, threshold=0.5):
    """
    改进版：
    同时奖励“填充结构”和“桥接区域”
    """

    gt_bin = (gt > threshold).astype(np.uint8)

    from scipy.ndimage import distance_transform_edt

    # 距离到结构
    dist_to_gt = distance_transform_edt(1 - gt_bin)

    # 权重：越靠近结构越重要
    weight = np.exp(-dist_to_gt / 3.0)

    # 加权响应
    score = np.sum(pred * weight) / np.sum(weight)

    return float(score)


def evaluate_one_epoch(gt_dir, pred_dir, num_images=10):
    l2_list, mse_list, rmse_list = [], [], []
    soft_conn_list = []

    for i in range(num_images):
        pred_name = f"test_pattern_{i:05d}.png"
        gt_name = f"resist_bottom_{i:05d}.png"

        pred_path = pred_dir / pred_name
        gt_path = gt_dir / gt_name

        if not gt_path.exists():
            print(f"[GT Missing] {gt_path}")
            continue

        if not pred_path.exists():
            print(f"[PRED Missing] {pred_path}")
            continue

        gt = load_image(gt_path)
        pred = load_image(pred_path)

        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch: {pred_name}")

        l2, mse, rmse = compute_metrics(gt, pred)
        soft_conn = compute_soft_connectivity_v2(gt, pred)

        l2_list.append(l2)
        mse_list.append(mse)
        rmse_list.append(rmse)
        soft_conn_list.append(soft_conn)

    if len(mse_list) == 0:
        return None

    return {
        "l2": np.mean(l2_list),
        "mse": np.mean(mse_list),
        "rmse": np.mean(rmse_list),
        "soft_conn": np.mean(soft_conn_list),
    }


def evaluate_all_epochs(exp_dir, gt_dir, max_epoch=100):
    results = []

    for epoch in range(1, max_epoch + 1):
        folder_name = f"infer_checkpoint_epoch_{epoch:03d}"
        pred_dir = exp_dir / folder_name   # ✅ 正确用 exp_dir

        if not pred_dir.exists():
            continue

        print(f"Processing epoch {epoch}...")

        metrics = evaluate_one_epoch(gt_dir, pred_dir)

        if metrics is None:
            continue

        results.append({
            "epoch": epoch,
            "l2": metrics["l2"],
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "soft_conn": metrics["soft_conn"],
        })

    return pd.DataFrame(results)


def plot_results(df, save_path):
    plt.figure()

    # 横坐标：1,2,3,...
    x = range(1, len(df) + 1)

    plt.plot(x, df["mse"], label="MSE")
    plt.plot(x, df["rmse"], label="RMSE")

    plt.xlabel("Epoch Index")
    plt.ylabel("Error")
    plt.title("Error vs Epoch")

    # 强制x轴为整数刻度
    plt.xticks(x)

    plt.legend()
    plt.grid()

    plt.savefig(save_path)
    plt.close()
    
def plot_soft_connectivity(df, save_path):
    plt.figure()

    x = range(1, len(df) + 1)

    plt.plot(x, df["soft_conn"], label="Soft Connectivity")

    plt.xlabel("Epoch Index")
    plt.ylabel("Soft Connectivity Score")
    plt.title("Soft Connectivity vs Epoch")

    plt.xticks(x)
    plt.legend()
    plt.grid()

    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # ===== 配置 =====
    exp_name = "exp_unet_edge_v2_bce_dice_0.5mse"
    exp_dir = project_root / "experiments" / exp_name
    gt_dir = project_root / "test_pattern" / "simulation_results" / "figure"
    max_epoch = 100

    # ===== 运行 =====
    df = evaluate_all_epochs(exp_dir, gt_dir, max_epoch)

    # ===== 保存 CSV =====
    csv_path = exp_dir / "evaluation_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # ===== 绘图 =====
    plot_path = exp_dir / "error_curve.png"
    plot_results(df, plot_path)
    print(f"Saved mse plot to {plot_path}")
    
    soft_plot_path = exp_dir / "soft_connectivity_curve.png"
    plot_soft_connectivity(df, soft_plot_path)
    print(f"Saved soft connectivity plot to {soft_plot_path}")