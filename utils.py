import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
import cv2
import os
import numpy as np


class SkyQualityModel:
    def __init__(self, model_path=None):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.mean = 14.77
        self.std = 0.65
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _load_model(self, model_path):
        class ImprovedSkyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_model = EfficientNet.from_pretrained("efficientnet-b0")
                self.dropout = nn.Dropout(0.2)
                self.fc = nn.Sequential(
                    nn.Linear(1280, 256), nn.ReLU(), nn.Linear(256, 1)
                )

            def forward(self, x):
                features = self.base_model.extract_features(x)
                features = nn.AdaptiveAvgPool2d(1)(features).flatten(1)
                features = self.dropout(features)
                return self.fc(features)

        model = ImprovedSkyModel().to(self.device)
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def preprocess(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img).unsqueeze(0).to(self.device)

    def predict(self, image_path):
        img_tensor = self.preprocess(image_path)
        with torch.no_grad():
            pred = self.model(img_tensor).item()
        return pred * self.std + self.mean  # Денормализация
