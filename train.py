import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

# Фикс для SSL
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class NormalizedSkyDataset(Dataset):
    def __init__(self, df, img_dir="data/images/", is_train=True):
        self.df = df
        self.img_dir = img_dir
        self.is_train = is_train

        # Нормализуем метки
        self.mean = df["NSB_mpsas"].mean()
        self.std = df["NSB_mpsas"].std()

        self.train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.is_train:
            img = self.train_transform(img)
        else:
            img = self.val_transform(img)

        label = (self.df.iloc[idx]["NSB_mpsas"] - self.mean) / self.std
        return img, torch.tensor(label, dtype=torch.float32)


class ImprovedSkyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = EfficientNet.from_pretrained("efficientnet-b0")
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Sequential(nn.Linear(1280, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        features = self.base_model.extract_features(x)
        features = nn.AdaptiveAvgPool2d(1)(features).flatten(1)
        features = self.dropout(features)
        return self.fc(features)


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv("data/metadata.csv")
    print("Original label stats:")
    print(f"Mean: {df['NSB_mpsas'].mean():.2f}, Std: {df['NSB_mpsas'].std():.2f}")
    print(f"Min: {df['NSB_mpsas'].min():.2f}, Max: {df['NSB_mpsas'].max():.2f}")

    # Удаление возможных проблемных данных
    df = df.dropna(subset=["NSB_mpsas"])

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = NormalizedSkyDataset(train_df, is_train=True)
    val_dataset = NormalizedSkyDataset(val_df, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = ImprovedSkyModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)

    # Gradient clipping
    max_grad_norm = 1.0
    best_rmse = float("inf")

    for epoch in range(50):
        model.train()
        train_loss = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            train_loss += loss.item()

        # Валидация
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs).squeeze()
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Денормализация для вычисления RMSE
        val_preds = np.array(val_preds) * train_dataset.std + train_dataset.mean
        val_labels = np.array(val_labels) * train_dataset.std + train_dataset.mean

        val_rmse = np.sqrt(np.mean((val_preds - val_labels) ** 2))
        train_loss /= len(train_loader)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), "model/best_model.pth")
            print(f"New best model saved! RMSE: {best_rmse:.4f}")

        scheduler.step(val_rmse)


if __name__ == "__main__":
    train()
