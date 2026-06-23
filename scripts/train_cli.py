import _init_paths
import os
import sys
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import NTIRETrainDataset, TransformSubset, get_transforms
from src.models import CLIPDetector, DINOv2Detector, TransFusionNet


try:
    from sklearn.metrics import roc_auc_score
except Exception as e:
    raise RuntimeError(
        "scikit-learn is required for roc_auc_score. Install with: pip install scikit-learn"
    ) from e


def parse_shards(s: str | None):
    """
    Helper to parse shard selection string into a list of integers or None (for all shards).
    Accepts:
      - None / "all" -> None (meaning: all shards)
      - "0" -> [0]
      - "0,1,2" -> [0,1,2]
    """
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in ("all", "none", ""):
        return None
    return [int(x) for x in s.split(",") if x.strip() != ""]


def build_model(model_name: str):
    """
    Helper to build model by name. Matches the models implemented in src/models.py
    Supported: clip, dino, transfusion
    """

    name = model_name.strip().lower()
    if name in ("clip", "clipdetector"):
        return CLIPDetector()
    if name in ("dino", "dinov2", "dinov2detector"):
        return DINOv2Detector()
    if name in ("transfusion", "transfusionnet", "fusion"):
        return TransFusionNet()
    raise ValueError(f"Unknown model '{model_name}'. Choose from: clip, dino, transfusion")


@torch.no_grad()
def evaluate_auc(model, loader, device, *, wandb_run=None, epoch: int | None = None):
    model.eval()
    all_probs = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=False)
        logits = model(images).squeeze()
        probs = torch.sigmoid(logits).detach().float().cpu().numpy()

        labels_np = labels.detach().cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels_np.tolist())

    auc = roc_auc_score(all_labels, all_probs)
    return auc


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def train(
    model,
    train_loader,
    val_loader,
    *,
    epochs,
    lr,
    device,
    accumulation_steps,
    weight_decay,
    checkpoint_dir,
    save_prefix,
    log_path,
    resume_path: str | None = None,
    wandb_run=None,
):
    #--------------------------------
    # SETUP: model, optimizer, scheduler, criterion, logging
    #--------------------------------
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)

    # Only optimize trainable parameters (important for frozen backbones / fusion-only training)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found. Check requires_grad flags.")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # Cosine scheduler (default), no CLI selection
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_auc": [], "epochs": []}
    best_auc = 0.0
    start_epoch = 0  # 0-based

    # Logging (append if resuming)
    log_f = open(log_path, "a" if resume_path else "w")

    def log(msg: str):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)
    log(f"Device: {device}")
    log(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    if resume_path:
        log(f"Resuming from: {resume_path}")
    log("=" * 60)

    #--------------------------------
    # RESUME CHECKPOINT (if provided)
    #--------------------------------
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            _move_optimizer_state_to_device(optimizer, device)

        

        if "history" in ckpt and ckpt["history"] is not None:
            history = ckpt["history"]

        # Track best auc (fallback to max over history if available)
        if "auc" in ckpt and ckpt["auc"] is not None:
            best_auc = float(ckpt["auc"])
        elif len(history.get("val_auc", [])) > 0:
            best_auc = float(max(history["val_auc"]))

        last_epoch = int(ckpt.get("epoch", -1))  # 0-based epoch stored in checkpoint
        start_epoch = last_epoch + 1

        # Ricreare lo scheduler sulle epoche rimanenti invece di caricare il vecchio state
        remaining_epochs = epochs - start_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)

        if start_epoch >= epochs:
            log(f"Checkpoint epoch={last_epoch} implies start_epoch={start_epoch}, but epochs={epochs}. Nothing to do.")
            log_f.close()
            return model, history

        log(f"✓ Loaded checkpoint (last_epoch={last_epoch}); continuing at epoch {start_epoch + 1}/{epochs}")

    #--------------------------------
    # TRAINING LOOP
    #--------------------------------

    global_step = 0

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        log(f"\nEpoch {epoch + 1}/{epochs}")
        log("-" * 60)

        # TRAIN
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=False)
            labels = labels.float().to(device, non_blocking=False)

            logits = model(images).squeeze()
            loss = criterion(logits, labels) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            batch_loss = float(loss.item() * accumulation_steps)
            train_loss += batch_loss
            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

            # LOG every train_loss in W&B
            if wandb_run is not None and global_step is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/loss_batch": float(batch_loss),
                    },
                    step=global_step,
                )
                global_step += 1

        avg_loss = train_loss / max(1, len(train_loader))
        val_auc = evaluate_auc(
            model,
            val_loader,
            device,
            wandb_run=wandb_run,
            epoch=epoch + 1
        )

        if wandb_run is not None:
            current_lr = float(optimizer.param_groups[0]["lr"])
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train/loss_epoch": float(avg_loss),
                    "val/roc_auc": float(val_auc),
                    "train/lr": current_lr,
                    "val/best_roc_auc": float(best_auc),
                },
                step=global_step,  # log at the current train-step
            )

        history["train_loss"].append(avg_loss)
        history["val_auc"].append(val_auc)
        history["epochs"].append(epoch + 1)

        epoch_time = time.time() - epoch_start
        remaining_epochs = epochs - (epoch + 1)
        remaining_time_h = (remaining_epochs * epoch_time) / 3600.0

        if wandb_run is not None:
            current_lr = float(optimizer.param_groups[0]["lr"])
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train/loss_epoch": float(avg_loss),
                    "val/roc_auc": float(val_auc),
                    "train/lr": current_lr,
                    "val/best_roc_auc": float(best_auc),
                },
                step=global_step,
            )

        log(f"Epoch {epoch+1}/{epochs} Results:")
        log(f"  Train Loss: {avg_loss:.4f}")
        log(f"  Val AUC:    {val_auc:.4f}")
        log(f"  Epoch time: {epoch_time/60:.1f} minutes")
        log(f"  Remaining:  ~{remaining_time_h:.1f} hours")

        if val_auc > best_auc:
            # Save best model checkpoint -> overwrite previous best
            best_ckpt_path = os.path.join(checkpoint_dir, f"{save_prefix}_best.pth")
            torch.save(
                {
                    "epoch": epoch,  # 0-based
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "auc": val_auc,
                    "loss": avg_loss,
                    "history": history,
                },
                best_ckpt_path,
            )
            best_auc = val_auc
            log("  🏆 New best AUC!")
        
        # Save checkpoint EVERY epoch
        ckpt_path = os.path.join(checkpoint_dir, f"{save_prefix}_epoch{epoch+1}.pth")
        torch.save(
            {
                "epoch": epoch,  # 0-based
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "auc": val_auc,
                "loss": avg_loss,
                "history": history,
            },
            ckpt_path,
        )
        log(f"  ✓ Checkpoint saved: {ckpt_path}")

        scheduler.step()

    log("\n" + "=" * 60)
    log(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Best AUC: {best_auc:.4f}")
    log("=" * 60)

    log_f.close()
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train CLIP/DINO/TransFusion models (merged notebooks CLI)")
    parser.add_argument(
        "--data-root",
        type=str,
        default="../data/train",
        help="Path to training data root (e.g., data/train)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dino",
        help="Model name: clip | dino | transfusion",
    )
    parser.add_argument(
        "--shards",
        type=str,
        default="all",
        help="Shard selection: all | 0 | 0,1,2 ...",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--save-prefix",
        type=str,
        default=None,
        help="Checkpoint prefix (default: model name)",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Log file path (default: checkpoints/<prefix>.log)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint .pth to resume from (loads model/optimizer/scheduler/history)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | mps | cuda",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (optional). Requires: pip install wandb",
    )
    parser.add_argument("--wandb-project", type=str, default="ntire2026-aigc-detection")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated tags, e.g. clip,aug448,shard0")
    parser.add_argument("--wandb-group", type=str, default=None)
    args = parser.parse_args()

    #--------------------------------
    # SETUP: device, reproducibility, model, dataset, logging
    #--------------------------------
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(repo_root)

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Reproducibility
    torch.manual_seed(args.seed)

    shards = parse_shards(args.shards)

    print("=" * 60)
    print("TRAINING (CLI)")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Model:  {args.model}")
    print(f"Shards: {shards if shards is not None else 'ALL'}")
    print("=" * 60)

    # Optional W&B init
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except Exception as e:
            raise RuntimeError(
                "W&B logging requested via --wandb but wandb is not installed. "
                "Install with: pip install wandb"
            ) from e

        tags = None
        if args.wandb_tags:
            tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            group=args.wandb_group,
            tags=tags,
            config=vars(args),
        )

    #--------------------------------
    # DATASET & MODEL
    #--------------------------------
    # Transforms (same pattern as notebook 09: base dataset has no transform)
    train_transform, val_transform = get_transforms()

    base_dataset = NTIRETrainDataset(
        args.data_root,
        shard_nums=shards,  # None -> all shards
        transform=None,
    )

    val_size = int(len(base_dataset) * args.val_split)
    train_size = len(base_dataset) - val_size

    # Split indices, then wrap with different transforms
    indices = list(range(len(base_dataset)))
    g = torch.Generator().manual_seed(args.seed)
    train_idx, val_idx = torch.utils.data.random_split(indices, [train_size, val_size], generator=g)

    train_dataset = TransformSubset(base_dataset, train_idx, train_transform)
    val_dataset = TransformSubset(base_dataset, val_idx, val_transform)

    print(f"Dataset split: Train={len(train_dataset):,} Val={len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = build_model(args.model)
    print(f"Creating model: {type(model).__name__}")

    save_prefix = args.save_prefix or args.model.lower()
    log_path = args.log_path or os.path.join(args.checkpoint_dir, f"{save_prefix}.log")

    #--------------------------------
    # TRAIN
    #--------------------------------
    model, history = train(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        accumulation_steps=args.accumulation_steps,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        save_prefix=save_prefix,
        log_path=log_path,
        resume_path=args.resume,
        wandb_run=wandb_run,
    )

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best val AUC: {max(history['val_auc']):.4f}")
    print(f"Log file: {log_path}")
    print(f"Checkpoints: {args.checkpoint_dir}/{save_prefix}_epoch*.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()