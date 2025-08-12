import pandas as pd
from tqdm import tqdm
from utils import SkyQualityModel


def predict():
    model = SkyQualityModel("model/best_model.pth")
    test_df = pd.read_csv("data/test.csv")

    predictions = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_path = f"data/images/{row['filename']}"
        try:
            pred = model.predict(img_path)
            predictions.append(pred)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            predictions.append(0.0)

    test_df["sky_quality"] = predictions
    test_df[["filename", "sky_quality"]].to_csv("outputs/submission.csv", index=False)


if __name__ == "__main__":
    predict()
