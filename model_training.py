# model_training.py

import os
import json
import argparse
from typing import List

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import joblib
import random

# -------------------------
# Dataset Class
# -------------------------
class FoodDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        image = Image.open(p).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx]).float()
        return image, label

# -------------------------
# Utilities
# -------------------------
def parse_ingredients_cell(s: str) -> List[str]:
    if pd.isna(s):
        return []
    if isinstance(s, list):
        return s
    s = str(s).strip()
    if s == "":
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts

def epoch_train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def epoch_val(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * imgs.size(0)

            preds = torch.sigmoid(outputs).cpu().numpy()
            

            all_preds.append((preds >= threshold).astype(int))
            all_targets.append(targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    micro_f1 = f1_score(all_targets, all_preds, average="micro", zero_division=0)
    micro_precision = precision_score(all_targets, all_preds, average="micro", zero_division=0)
    micro_recall = recall_score(all_targets, all_preds, average="micro", zero_division=0)
    return running_loss / len(loader.dataset), micro_f1, micro_precision, micro_recall

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Train ingredient multi-label model")
    parser.add_argument("--csv", type=str, default="completed_dataset.csv")
    parser.add_argument("--images", type=str, default="food_images")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="model_artifacts")
    parser.add_argument("--img_ext", type=str, default=".jpg")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    CSV_PATH = args.csv
    IMAGES_DIR = args.images
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    OUT_DIR = args.output_dir
    IMG_EXT = args.img_ext
    SEED = args.seed

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    # -------------------------
    # Load CSV
    # -------------------------
    print("Loading CSV:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    for col in ["Image_Path", "Ingredients"]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column `{col}` in CSV")

    rows = []
    miss_img = 0
    for _, r in df.iterrows():
        img_key = str(r["Image_Path"]).strip()
        img_paths_to_try = [
            os.path.join(IMAGES_DIR, img_key + IMG_EXT),
            os.path.join(IMAGES_DIR, img_key),
            os.path.join(IMAGES_DIR, img_key + ".jpeg"),
            os.path.join(IMAGES_DIR, img_key + ".png"),
        ]
        found = next((p for p in img_paths_to_try if os.path.exists(p)), None)
        if not found:
            miss_img += 1
            continue
        ing_list = parse_ingredients_cell(r["Ingredients"])
        if len(ing_list) == 0:
            continue
        rows.append((found, ing_list))

    print(f"Total rows with images & ingredients: {len(rows)} (skipped {miss_img} missing images)")

    # -------------------------
    # Prepare labels
    # -------------------------
    all_ingredients = sorted({ing.strip() for _, ing_list in rows for ing in ing_list if ing.strip()})
    print(f"Unique ingredient tokens: {len(all_ingredients)}")
    mlb = MultiLabelBinarizer(classes=all_ingredients)
    y_all = mlb.fit_transform([ing_list for _, ing_list in rows])

    with open(os.path.join(OUT_DIR, "ingredients.json"), "w", encoding="utf-8") as f:
        json.dump(all_ingredients, f, indent=2)
    joblib.dump(mlb, os.path.join(OUT_DIR, "mlb.pkl"))
    print("Saved mlb.pkl and ingredients.json in", OUT_DIR)

    # -------------------------
    # Train/Val split
    # -------------------------
    image_paths = [p for p, _ in rows]
    labels = y_all
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.1, random_state=SEED
    )
    print(f"Train size: {len(train_paths)}, Val size: {len(val_paths)}")

    # -------------------------
    # DataLoader
    # -------------------------
    IMG_SIZE = 224
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_ds = FoodDataset(train_paths, train_labels, transform=train_transform)
    val_ds = FoodDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    # -------------------------
    # Model
    # -------------------------
    num_classes = len(all_ingredients)
    print("Num classes (ingredients):", num_classes)

    try:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except Exception:
        model = models.resnet50(weights='DEFAULT')

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # -------------------------
    # Loss, optimizer, scheduler
    # -------------------------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # -------------------------
    # Training loop
    # -------------------------
    best_val_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = epoch_train(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1, val_prec, val_rec = epoch_val(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{EPOCHS} — train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
              f"val_f1: {val_f1:.4f} | prec: {val_prec:.4f} | rec: {val_rec:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_path = os.path.join(OUT_DIR, "best_food_model.pth")
            torch.save(model.state_dict(), best_path)
            print("Saved best model to:", best_path)

    final_model_path = os.path.join(OUT_DIR, "food_model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print("Saved final model to:", final_model_path)

    meta = {
        "num_samples": len(rows),
        "num_classes": num_classes,
        "ingredients_file": os.path.join(OUT_DIR, "ingredients.json"),
        "mlb_file": os.path.join(OUT_DIR, "mlb.pkl"),
        "model_file": final_model_path
    }
    with open(os.path.join(OUT_DIR, "train_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved training metadata.")

if __name__ == "__main__":
    main()
