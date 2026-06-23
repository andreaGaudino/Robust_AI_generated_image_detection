import os
import sys
import argparse
from datetime import datetime

import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Make repo imports work when running from scripts/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from src.models import CLIPDetector, DINOv2Detector, TransFusionNet


class TestDataset(Dataset):
    """Load test images without labels"""

    def __init__(self, images_dir: str, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        self.image_names = sorted(
            [
                f
                for f in os.listdir(images_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        if len(self.image_names) == 0:
            raise FileNotFoundError(f"No images found in: {images_dir}")

        print(f"Loaded {len(self.image_names)} images from: {images_dir}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = np.zeros((448, 448, 3), dtype=np.uint8)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, img_name


def build_model(model_type: str):
    mt = model_type.strip().lower()
    if mt == "clip":
        return CLIPDetector()
    if mt == "dino":
        return DINOv2Detector()
    if mt == "transfusion":
        return TransFusionNet()
    raise ValueError("Unknown --model. Choose from: clip | dino | transfusion")


def auto_device(requested: str | None) -> str:
    if requested and requested.lower() != "auto":
        return requested.lower()
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Inference CLI for NTIRE 2026 AIGC Detection")
    parser.add_argument("--model", type=str, required=True, help="clip | dino | transfusion")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--images-dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output-csv", type=str, required=True, help="Output CSV path (image_name,label)")
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = auto_device(args.device)

    print("=" * 60)
    print("INFERENCE (CLI)")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device:  {device}")
    print(f"Model:   {args.model}")
    print(f"CKPT:    {args.model_path}")
    print(f"Images:  {args.images_dir}")
    print("=" * 60)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")
    if not os.path.isdir(args.images_dir):
        raise FileNotFoundError(f"Images dir not found: {args.images_dir}")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    # Transforms (same as your inference.py / notebook)
    test_transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=512, interpolation=1),
            A.CenterCrop(448, 448),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    # Dataset / loader
    ds = TestDataset(args.images_dir, transform=test_transform)
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # Model
    model = build_model(args.model)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    if isinstance(ckpt, dict) and "auc" in ckpt:
        print(f"✓ Checkpoint val AUC: {ckpt['auc']:.4f}")

    # Predict
    results = []
    with torch.no_grad():
        for images, img_names in tqdm(dl, desc="Predicting"):
            images = images.to(device)
            logits = model(images).view(-1)
            probs = torch.sigmoid(logits).detach().float().cpu().numpy()

            for name, p in zip(img_names, probs.tolist()):
                results.append({"image_name": name, "label": float(p)})

    df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"✓ Saved: {args.output_csv}")
    print(f"  N={len(df):,}")
    print(f"  Mean={df['label'].mean():.4f}  Std={df['label'].std():.4f}")
    print(f"  Min={df['label'].min():.4f}  Max={df['label'].max():.4f}")
    print(
        f"  Fake(>0.5)={(df['label'] > 0.5).sum():,}  Real(<=0.5)={(df['label'] <= 0.5).sum():,}"
    )

    out_dir = os.path.dirname(args.output_csv) or "."
    out_base = os.path.basename(args.output_csv)
    codabench_path = os.path.join(out_dir, f"codabench_{out_base}")

    os.makedirs(os.path.dirname(codabench_path) or ".", exist_ok=True)

    df_codabench = df.rename(columns={"label": "score"})
    df_codabench.to_csv(codabench_path, index=False)
    print(f"✓ CodaBench CSV: {codabench_path}")


if __name__ == "__main__":
    main()