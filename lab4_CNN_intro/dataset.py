import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms as tfms


class AnimalsDataset(Dataset):
    def __init__(self, csv_file, transforms=None):
        # read the csv file
        self.length = len(csv_file)
        self.filepaths = csv_file[["file"]].to_numpy().flatten()
        self.labels = csv_file[["label"]].to_numpy().flatten()

        # save transforms for later
        if transforms is None:
            transforms = tfms.Compose([
                tfms.ToTensor(),
                tfms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.transforms = transforms

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        img = Image.open(filepath)
        img = self.transforms(img)

        label = self.labels[index]

        return img, label
