from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
#from hydra.utils import instantiate
import torch
from torch.utils.data import Dataset,Subset, ConcatDataset
# import image
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

class DataModule:
    def __init__(
        self,
        train_dataset_path,
        unlabeled_dataset_path,
        train_transform,
        batch_size,
        num_workers,
        top_k=15,
        top_p=5,
        threshold=0,
    ):
        self.num_classes = len(os.listdir(train_dataset_path))
        self.labeled_dataset = ImageFolder(train_dataset_path, transform=train_transform)
        # unlabeled dataset is a dataset with no labels so it is not an ImageFolder.
        # load images from unlabeled_dataset_path and apply train_transform without calling ImageFolder
        self.unlabelled_dataset = UnlabeledDataset(unlabeled_dataset_path, train_transform, batch_size, num_workers)
        # for debugging purposes, we only use a small subset of the unlabeled dataset
        self.unlabelled_dataset = Subset(self.unlabelled_dataset, list(range(0, 10000)))
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.already_labeled = []
        self.remaining = list(range(len(self.unlabelled_dataset)))
        self.top_k = top_k
        self.top_p = top_p
        self.threshold = threshold

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
    
    def generate_folds(self, n_folds=5,shuffle=True):
        # split the dataset into n_folds for cross validation
        # return a list of n_folds dataset pairs (train_set, val_set)
        split = len(self.labeled_dataset) // n_folds
        folds = []
        for i in range(n_folds):
            train_loader = DataLoader(
                Subset(self.labeled_dataset,
                            list(range(0, i*split)) + list(range((i+1)*split, len(self.labeled_dataset)))),
                batch_size=self.batch_size, 
                shuffle=shuffle, 
                num_workers=self.num_workers
                )
            val_loader = DataLoader(
                Subset(self.labeled_dataset, 
                            list(range(i*split, (i+1)*split))),
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers
                )
            
            folds.append((train_loader, val_loader))
        return folds
    
    def print_class_distribution(self, output_file=None):
        # print the class distribution of the dataset
        _, counts = np.unique(self.labeled_dataset.targets, return_counts=True)
        print("Class distribution:")
        # plot the histogram 
        plt.bar(np.arange(len(counts)), counts)
        plt.xticks(np.arange(len(counts)))
        plt.xlabel("Class")
        plt.ylabel("Count")
        if output_file is not None:
            plt.savefig(output_file)
        plt.show()
        return
    
    def add_labels(self, model, pseudo_label_loader, device):
        # add labels to the unlabeled dataset
        # already_labeled: list of indices of images that have already been labeled
        # remaining: list of indices of images that have not been labeled yet
        # return a new dataloader with the newly labeled images

        # get the predictions of the model on the unlabeled dataset
        model.to(device)
        model.eval()
        if self.remaining == []:
            return pseudo_label_loader
        
        remaining_loader = DataLoader(Subset(self.unlabelled_dataset, self.remaining), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        scores, classes = torch.empty((0, self.top_p), device=device), torch.empty((0, self.top_p), device=device)

        with torch.no_grad():
            for batch in tqdm(remaining_loader):
                batch = batch.to(device)
                output = model(batch)
                output = torch.softmax(output, dim=1)
                # for each image, get the top P predictions
                score, pred = torch.topk(output, self.top_p, dim=1)
                score.to(device)
                pred.to(device)

                
                scores = torch.cat((scores, score), dim=0)
                classes = torch.cat((classes, pred), dim=0)
        assert scores.shape == (len(self.remaining), self.top_p)
        # scores, classes have shape (len(remaining), self.top)
        # get the top K predictions for each class
        print(f"classes shape: {classes.shape}")
        print(classes)
        by_classes = torch.zeros((self.num_classes, self.top_k, 2), device=device)
        for i in range(self.num_classes):
            # get the indices of all images with class i in their top self.top
            idx = torch.where(classes.int() == i)
            if idx[0].shape[0] == 0:
                continue
            # get the scores of all images with class i in their top self.top
            
            score = scores[idx]
            class_i = torch.stack((idx[0], score), dim=1)
            # Sort the scores in descending order
            class_i = class_i[class_i[:,1].argsort(descending=True)]
            # get the top K scores
            class_i = class_i[:self.top_k]
            # pad with 0 if there are less than K images with class i in their top self.top
            if class_i.shape[0] < self.top_k:
                class_i = torch.cat((class_i, torch.zeros((self.top_k - class_i.shape[0], 2), device=device)), dim=0)

            # class_i has shape (self.top_k, 2)
            print(class_i.shape)
            assert class_i.shape == (self.top_k, 2)
            by_classes[i] = class_i
        
        # by_classes has shape (num_classes, self.top_k, 2)
        # linearize into a tensor of shape (num_classes * self.top_k, 3) with entries (class, position, score)
        class_id  = torch.arange(self.num_classes).unsqueeze(-1).unsqueeze(-1).repeat(1, self.top_k, 1).reshape(self.num_classes, self.top_k, 1).to(device)
        by_classes = torch.cat((class_id,by_classes), dim=2)
        by_classes = by_classes.reshape(-1, 3)

        if self.threshold > 0:
            # filter by threshold
            by_classes = by_classes[by_classes[:,2] > self.threshold]
        # remove the entries with zero score
        by_classes = by_classes[by_classes[:,2] > 0]
        # update the already labeled and remaining lists, making sure to cast floating point indices to integers
        indices = by_classes[:,1].int().tolist()
        self.already_labeled = list(set(self.already_labeled) | set(indices))
        self.remaining = list(set(self.remaining) - set(self.already_labeled))

        # create a new dataset with the newly labeled images
        # remaining_loader.dataset[i] returns the i-th image in the remaining dataset, i.e. a tensor.
        assert len(indices) > 0 
        print(f"indices: {indices}")
        
        images = [remaining_loader.dataset[i] for i in indices]
        images = torch.stack(images, dim=0)
        targets = by_classes[:,0].int()
        print(f"targets: {targets}")
        assert targets.shape == (len(indices),)
<<<<<<< HEAD
        images = images.cpu()
        targets = targets.cpu()
        
=======
        # send images and targets back to the cpu for storage
        images = images.cpu()
        targets = targets.cpu()

>>>>>>> 0dc36867730f668243a3f4bee02ceed74d00bc35
        new_dataset = torch.utils.data.TensorDataset(images, targets)
        # create a new dataloader with the newly labeled images
        if pseudo_label_loader is not None:
            # append the new dataloader to the pseudo_label_loader
            new_dataset = ConcatDataset([pseudo_label_loader.dataset, new_dataset])
        new_loader = DataLoader(new_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return new_loader

    def reset_labels(self):
        # reset the labels of the dataset
        self.already_labeled = []
        self.remaining = list(range(len(self.unlabelled_dataset)))
        return    

    def label_all(self, model, train_loader, device):
        # label all the images in the unlabeled dataset
        # return a new dataloader with all the images
        loader = DataLoader(self.unlabelled_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        scores, classes = torch.Tensor([]), torch.Tensor([])
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                output = model(batch)
                # for each image, get the top prediction
                _, pred = torch.max(output, dim=1, keepdim=True)
                # pred has shape (batch_size, 1)
                classes = torch.cat((classes, pred), dim=0)
        # classes has shape (len(unlabelled_dataset), 1)
        # create a new dataset with the newly labeled images
        images = [loader.dataset[i] for i in range(len(loader.dataset))]
        images = torch.stack(images, dim=0)
        targets = classes.int()
        targets = torch.stack(targets, dim=0)
        new_dataset = torch.utils.data.TensorDataset(images, targets)
        new_dataset = ConcatDataset([train_loader.dataset, new_dataset])
        new_loader = DataLoader(new_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return new_loader





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
    

if __name__ == "__main__":
    a = torch.randn(4, 4)
    print(a)
    b = torch.argsort(a, dim=0)
    print(b)
    # sort the first row of a in descending order
    print(b[:,0])
    print(a[b[:,0]])
    
    