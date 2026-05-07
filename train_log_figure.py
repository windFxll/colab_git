import re
import matplotlib.pyplot as plt
from pathlib import Path


def extract_avg_loss(log_path):
    """
    从日志文件中提取 Avg Loss 数值
    """
    avg_losses = []

    # 正则：匹配 Avg Loss = 后面的数字（支持科学计数法）
    pattern = re.compile(r"Avg Loss\s*=\s*([0-9.eE+-]+)")

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                loss = float(match.group(1))
                avg_losses.append(loss)

    return avg_losses


def plot_loss(avg_losses):
    """
    画二维散点图
    """
    epochs = list(range(1, len(avg_losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.scatter(epochs, avg_losses)

    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss")
    plt.title("Training Avg Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from pathlib import Path

    # ======== 指定 logs 目录（不用写具体文件名）========
    logs_dir = Path("experiments/exp_unet_edge_v2_bce_dice_0.5sobel/logs")

    # ======== 找所有 train_*.log ========
    log_files = sorted(logs_dir.glob("train_*.log"))

    if len(log_files) == 0:
        print("logs 目录下没有找到 train_*.log 文件")
        exit()

    # ======== 取最新的一个（按文件名排序）========
    log_file = log_files[-1]
    print(f"使用日志文件: {log_file}")

    avg_losses = extract_avg_loss(log_file)

    if len(avg_losses) == 0:
        print("⚠️ 没有找到 Avg Loss，请检查日志格式")
    else:
        print(f"读取到 {len(avg_losses)} 个点")

        # ======== 保存路径 ========
        save_path = log_file.parent / f"{log_file.stem}_loss.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # ======== 画图 ========
        epochs = list(range(1, len(avg_losses) + 1))

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.scatter(epochs, avg_losses)
        plt.plot(epochs, avg_losses)

        plt.xlabel("Epoch")
        plt.ylabel("Avg Loss")
        plt.title("Training Avg Loss")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"✅ 图已保存到: {save_path}")