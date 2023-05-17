from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from hydra.utils import instantiate


class DataModule:
    def __init__(
        self,
        train_dataset_path,
        val_dataset_path,
        train_transform,
        val_transform,
        batch_size,
        num_workers,
    ):
        print(train_dataset_path, val_dataset_path)
        self.train_dataset = ImageFolder(train_dataset_path, transform=train_transform)
        self.val_dataset = ImageFolder(val_dataset_path, transform=val_transform)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
