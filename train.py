import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from utils import SkyQualityModel


class SkyQualityDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = f"data/images/{self.df.iloc[idx]['filename']}"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        label = self.df.iloc[idx]["NSB_mpsas"]
        return img, torch.tensor(label, dtype=torch.float32)


def train():
    device = torch.device("cpu")
    df = pd.read_csv("data/metadata.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = SkyQualityDataset(train_df, transform)
    val_dataset = SkyQualityDataset(val_df, transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = SkyQualityModel().model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(10):
        model.train()
        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Валидация
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs).squeeze()
                val_loss += criterion(outputs, labels).item()
        print(f"Epoch {epoch}, Val Loss: {val_loss / len(val_loader)}")

    torch.save(model.state_dict(), "models/efficientnet_b0.pth")


if __name__ == "__main__":
    train()
