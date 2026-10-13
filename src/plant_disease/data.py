from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SMALLCNN_MEAN = (0.5, 0.5, 0.5)
SMALLCNN_STD = (0.5, 0.5, 0.5)


def load_split(dataset_root: Path, split: str):
    """Load train/validation rows from the development manifest only."""
    if split not in {"train", "validation"}:
        raise ValueError("This comparison tool accepts train or validation. Use the final-evaluation notebook for test data.")
    root = Path(dataset_root)
    manifest = root / "data" / "manifests" / "plantvillage_development.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing development manifest: {manifest}")
    frame = pd.read_csv(manifest)
    required = {"split", "relative_path", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(frame.columns))}")
    frame = frame.loc[frame["split"] == split].reset_index(drop=True)
    classes = sorted(frame["label"].dropna().unique().tolist())
    if frame.empty or not classes:
        raise ValueError(f"No rows found for split={split!r} in {manifest}")
    return frame, classes


class PlantVillageImages(Dataset):
    def __init__(self, dataset_root: Path, frame: pd.DataFrame, classes: list[str], model_name: str, image_size=224):
        self.image_root = Path(dataset_root) / "data" / "raw" / "PlantVillage" / "train"
        self.rows = frame.reset_index(drop=True)
        self.label_to_index = {label: index for index, label in enumerate(classes)}
        self.model_name = model_name
        mean, std = (SMALLCNN_MEAN, SMALLCNN_STD) if model_name == "smallcnn" else (IMAGENET_MEAN, IMAGENET_STD)
        self.transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        path = self.image_root / row["relative_path"]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.label_to_index[row["label"]]
