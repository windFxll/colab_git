import argparse
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import yaml

import logging
from datetime import datetime

from datasets import LithoDataset
from losses import CombinedLoss
from models.registry import MODEL_REGISTRY


def setup_logger(output_dir):
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"train.log"

    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler（很重要）
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 写入文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # 输出到终端
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger, log_file

def save_config(config, log_dir):
    config_path = log_dir / f"config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)
    return config_path

def parse_args():
    parser = argparse.ArgumentParser(description="Lithography model training entrypoint.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_unet.yaml",
        help="Path to yaml config file.",
    )
    return parser.parse_args()


def load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name):
    key = str(model_name).lower()

    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[key]()


def main():
    args = parse_args()
    config = load_config(args.config)
    criterion = CombinedLoss(config["loss"])
    project_root = Path(__file__).resolve().parent

    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = resolve_device(config.get("device", "auto"))
    data_root = project_root / config.get("data_root", "data")
    pattern_types = config["pattern_types"]
    epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    log_every = int(config.get("log_every", 1))
    num_workers = int(config.get("num_workers", 0))
    lr = float(config["lr"])

    output_root = Path(config["output_root"])
    output_dir = output_root / config["output_dir"]
    
    output_dir.mkdir(parents=True, exist_ok=True)

    logger, log_file = setup_logger(output_dir)
    
    log_dir = output_dir / "logs"
    config_path = save_config(config, log_dir)

    logger.info(f"[Init] Config saved to: {config_path}")
    
    logger.info(f"Log file: {log_file}")
    
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = LithoDataset(str(data_root), pattern_types)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    logger.info(f"[Init] Config: {Path(args.config).resolve()}")
    logger.info(f"[Init] Device: {device}")
    logger.info(f"[Init] Data root: {data_root}")
    logger.info(f"[Init] Dataset size: {len(dataset)}")
    logger.info(f"[Init] Batches/epoch: {len(loader)}")

    model = build_model(config["model"]).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )

    best_loss = float("inf")
    last_completed_epoch = -1
    last_epoch_loss = None

    def save_checkpoint(path, epoch_idx, epoch_loss, is_interrupt=False):
        torch.save(
            {
                "epoch": epoch_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch_loss": epoch_loss,
                "is_interrupt": is_interrupt,
            },
            str(path),
        )

    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            logger.info(f"\n[Epoch {epoch + 1}/{epochs}] Start")

            for batch_idx, (x, y, weight) in enumerate(loader, start=1):
                x = x.to(device)
                y = y.to(device)
                weight = weight.to(device)

                pred = model(x)
                loss, loss_dict = criterion(pred, y, weight)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                if batch_idx % log_every == 0:
                    log_str = " ".join([f"{k}:{v:.4f}" for k, v in loss_dict.items()])
                    logger.info(
                        f"[Epoch {epoch+1}] Batch {batch_idx}/{len(loader)} "
                        f"Total={loss.item():.4f} {log_str}"
                    )

            epoch_loss = total_loss / len(loader)
            logger.info(f"[Epoch {epoch + 1}/{epochs}] Avg Loss = {epoch_loss:.6f}")

            save_every = int(config.get("save_every", 5))

            if (epoch + 1) % save_every == 0:
                epoch_ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
                save_checkpoint(epoch_ckpt_path, epoch, epoch_loss)
                logger.info(f"[Save] Epoch checkpoint -> {epoch_ckpt_path}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_ckpt_path = checkpoint_dir / "best_model.pt"
                save_checkpoint(best_ckpt_path, epoch, best_loss)
                logger.info(f"[Save] Best model updated (loss={best_loss:.6f}) -> {best_ckpt_path}")

            last_completed_epoch = epoch
            last_epoch_loss = epoch_loss

    except KeyboardInterrupt:
        interrupt_ckpt_path = checkpoint_dir / "interrupt_checkpoint.pt"
        save_checkpoint(
            interrupt_ckpt_path,
            last_completed_epoch,
            last_epoch_loss,
            is_interrupt=True,
        )
        logger.info(f"\n[Interrupt] Training interrupted. Checkpoint saved to {interrupt_ckpt_path}")
        
        raise
    finally:
        final_model_path = checkpoint_dir / "last_model.pt"
        torch.save(model.state_dict(), str(final_model_path))
        logger.info(f"[Done] Last model weights saved to {final_model_path}")


if __name__ == "__main__":
    main()