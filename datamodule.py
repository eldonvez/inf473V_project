from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
#from hydra.utils import instantiate
import torch
from torch.utils.data import Dataset
# import image
import os
from PIL import Image

class DataModule:
    def __init__(
        self,
        train_dataset_path,
        unlabeled_dataset_path,
        train_transform,
        batch_size,
        num_workers,
    ):
        self.labeled_dataset = ImageFolder(train_dataset_path, transform=train_transform)
        # unlabeled dataset is a dataset with no labels so it is not an ImageFolder.
        # load images from unlabeled_dataset_path and apply train_transform without calling ImageFolder
        self.unlabelled_dataset = UnlabeledDataset(unlabeled_dataset_path, train_transform, batch_size, num_workers)
        self.num_workers = num_workers
        self.batch_size = batch_size

    def dloader_labeled(self):
        return DataLoader(
            self.labeled_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    def dloader_unlabeled(self):
        return DataLoader(
            self.unlabelled_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

class UnlabeledDataset(Dataset):
    def __init__(self, dataset_path, transform, batch_size, num_workers):
        self.dataset_path = dataset_path
        self.transform = transform
        self.images = os.listdir(dataset_path)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dataset_path, self.images[idx]))
        img = self.transform(img)
        return img
    