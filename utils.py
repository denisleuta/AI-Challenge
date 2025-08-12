import cv2
import torch
import numpy as np
from torchvision import transforms
from efficientnet_pytorch import EfficientNet


class SkyQualityModel:
    def __init__(self, model_path=None):
        self.device = torch.device("cpu")
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=1)
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model.to(self.device)

    def preprocess(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return transform(img).unsqueeze(0).to(self.device)

    def predict(self, image_path):
        img_tensor = self.preprocess(image_path)
        with torch.no_grad():
            pred = self.model(img_tensor).item()
        return pred
