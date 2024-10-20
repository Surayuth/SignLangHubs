import cv2
from pathlib import Path
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


# Need to create the base class for this too
class SimpleDataset(Dataset):
    # see https://albumentations.ai/docs/examples/pytorch_classification/
    def __init__(self, paths, mapper, transform=None):
        self.paths = paths
        self.mapper = mapper
        self.labels = [self.mapper[Path(p).parent.name] for p in paths]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        label = self.labels[idx]
        return image, label
