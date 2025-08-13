import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
import numpy as np
import os
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
import joblib
import warnings

warnings.filterwarnings("ignore")

# Fix for SSL
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=25, delta=0.0005):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
        return self.early_stop


# Dataset
class AdvancedSkyDataset(Dataset):
    def __init__(self, df, img_dir="data/images/", is_train=True, mean=None, std=None):
        self.df = df
        self.img_dir = img_dir
        self.is_train = is_train
        if mean is None or std is None:
            raise ValueError("Mean and std must be provided")
        self.mean = mean
        self.std = std
        if self.std == 0:
            raise ValueError("Standard deviation of NSB_mpsas is zero")

        self.train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx]["filename"])
        if not os.path.exists(img_path):
            print(f"Warning: Image not found {img_path}")
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to load image {img_path}")
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.is_train:
            img = self.train_transform(img)
        else:
            img = self.val_transform(img)

        label = (self.df.iloc[idx]["NSB_mpsas"] - self.mean) / self.std
        if np.isnan(label) or np.isinf(label):
            print(f"Invalid label at index {idx}: {label}")
            label = 0.0
        return img, torch.tensor(label, dtype=torch.float32)


# Model
class AdvancedSkyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = EfficientNet.from_pretrained("efficientnet-b0")
        # Unfreeze last 4 blocks initially
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.base_model._blocks[-4:].parameters():
            param.requires_grad = True
        feature_dim = 1280  # For efficientnet-b0
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        features = self.base_model.extract_features(x)
        features = nn.AdaptiveAvgPool2d(1)(features).flatten(1)
        return self.regressor(features)


# Loss Function
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return F.smooth_l1_loss(output, target)


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and clean data
    df = pd.read_csv("data/metadata.csv")
    df = df.dropna(subset=["NSB_mpsas"])
    df = df[~df["NSB_mpsas"].isin([np.inf, -np.inf])]

    # Remove outliers using IQR
    Q1 = df["NSB_mpsas"].quantile(0.25)
    Q3 = df["NSB_mpsas"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["NSB_mpsas"] >= Q1 - 1.5 * IQR) & (df["NSB_mpsas"] <= Q3 + 1.5 * IQR)]

    if len(df) < 100:
        print(f"Warning: Small dataset size ({len(df)}), RMSE < 0.2 may be difficult")

    print("Data statistics:")
    print(f"Size: {len(df)}")
    print(f"Mean: {df['NSB_mpsas'].mean():.4f}")
    print(f"Std: {df['NSB_mpsas'].std():.4f}")
    print(f"Min: {df['NSB_mpsas'].min():.4f}")
    print(f"Max: {df['NSB_mpsas'].max():.4f}")

    # Stratified split
    df["bin"] = pd.cut(df["NSB_mpsas"], bins=5)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["bin"]
    )
    train_df = train_df.drop("bin", axis=1)
    val_df = val_df.drop("bin", axis=1)

    train_mean = train_df["NSB_mpsas"].mean()
    train_std = train_df["NSB_mpsas"].std()
    train_dataset = AdvancedSkyDataset(
        train_df, is_train=True, mean=train_mean, std=train_std
    )
    val_dataset = AdvancedSkyDataset(
        val_df, is_train=False, mean=train_mean, std=train_std
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=2)

    model = AdvancedSkyModel().to(device)
    criterion = CustomLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-5, total_steps=50 * len(train_loader), pct_start=0.3
    )
    early_stopping = EarlyStopping(patience=25, delta=0.0005)

    best_rmse = float("inf")
    history = {"train_loss": [], "val_rmse": []}
    grad_accum_steps = 4  # Simulate larger batch size

    # Two-stage training: partial unfreeze, then full unfreeze
    for stage in range(2):
        if stage == 1:
            print("Unfreezing entire backbone for fine-tuning")
            for param in model.base_model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=5e-6, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=5e-6,
                total_steps=50 * len(train_loader),
                pct_start=0.3,
            )

        for epoch in range(50):
            model.train()
            train_loss = 0
            optimizer.zero_grad()

            for i, (imgs, labels) in enumerate(
                tqdm(train_loader, desc=f"Stage {stage}, Epoch {epoch}")
            ):
                imgs, labels = imgs.to(device), labels.to(device)

                if (
                    torch.isnan(imgs).any()
                    or torch.isinf(imgs).any()
                    or torch.isnan(labels).any()
                    or torch.isinf(labels).any()
                ):
                    print(f"Skipping batch {i} due to invalid inputs")
                    continue

                outputs = model(imgs).squeeze()
                loss = criterion(outputs, labels) / grad_accum_steps
                if torch.isnan(loss):
                    print(f"NaN loss in batch {i}")
                    continue

                loss.backward()
                if (i + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                train_loss += loss.item() * grad_accum_steps

            train_loss /= len(train_loader)
            model.eval()
            val_preds, val_labels = [], []

            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs).squeeze()
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            val_preds = np.array(val_preds) * train_std + train_mean
            val_labels = np.array(val_labels) * train_std + train_mean

            if np.any(np.isnan(val_preds)) or np.any(np.isnan(val_labels)):
                print("NaN in validation predictions or labels")
                continue

            val_rmse = np.sqrt(np.mean((val_preds - val_labels) ** 2))
            history["train_loss"].append(train_loss)
            history["val_rmse"].append(val_rmse)

            print(
                f"Stage {stage}, Epoch {epoch}: Train Loss: {train_loss:.6f}, Val RMSE: {val_rmse:.6f}"
            )

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                torch.save(model.state_dict(), "model/best_model.pth")
                print(f"New best model saved! RMSE: {best_rmse:.6f}")

                plt.figure(figsize=(10, 6))
                plt.scatter(val_labels, val_preds, alpha=0.5)
                plt.plot(
                    [min(val_labels), max(val_labels)],
                    [min(val_labels), max(val_labels)],
                    "r--",
                )
                plt.xlabel("True Values")
                plt.ylabel("Predictions")
                plt.title(f"Predictions vs True (RMSE: {val_rmse:.6f})")
                plt.savefig("outputs/predictions.png")
                plt.close()

            if val_rmse < 0.2:
                print("Target RMSE < 0.2 achieved!")
                break

            if early_stopping(val_rmse):
                print("Early stopping triggered")
                break

        if val_rmse < 0.2:
            break

    # Calibration
    print("Calibrating model...")
    model.load_state_dict(torch.load("model/best_model.pth", map_location=device))
    model.eval()

    val_preds, val_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs).squeeze()
            val_preds.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_preds = np.array(val_preds) * train_std + train_mean
    val_labels = np.array(val_labels) * train_std + train_mean

    if np.any(np.isnan(val_preds)) or np.any(np.isnan(val_labels)):
        print("NaN in calibration data")
        return

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_preds, val_labels)
    joblib.dump(calibrator, "model/calibrator.pkl")

    calibrated_preds = calibrator.transform(val_preds)
    calibrated_rmse = np.sqrt(np.mean((calibrated_preds - val_labels) ** 2))
    print(f"RMSE after calibration: {calibrated_rmse:.6f}")

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["val_rmse"], label="Val RMSE")
    plt.title("Validation RMSE")
    plt.legend()

    plt.tight_layout()
    plt.savefig("outputs/training_history.png")
    plt.close()


if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    train()
