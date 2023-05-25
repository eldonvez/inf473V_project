from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import imp
from tqdm import tqdm
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt

def generate_folds(dataset, n_folds, batch_size=1, shuffle=True, num_workers=1):
    # split the dataset into n_folds for cross validation
    # return a list of n_folds dataset pairs (train_set, val_set)
    split = len(dataset) // n_folds
    folds = []
    for i in range(n_folds):
        train_loader = data.DataLoader(
            data.Subset(dataset,
                         list(range(0, i*split)) + list(range((i+1)*split, len(dataset)))),
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
            )
        val_loader = data.DataLoader(
            data.Subset(dataset, 
                        list(range(i*split, (i+1)*split))),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
            )
        
        folds.append((train_loader, val_loader))
    return folds

def balance_data(dataset, batch_size=1, shuffle=True, num_workers=1):
    # balance the dataset by oversampling the minority classes
    # return a dataloader, and the expansion factor
    _, counts = np.unique(dataset.targets, return_counts=True)
    lcm = np.lcm.reduce(counts)
    num_classes = len(counts)
    num_samples = lcm // counts
    index =  []
    for i in range(len(dataset)):
        index.extend([i] * num_samples[dataset.targets[i]])
    balanced_dataset = data.Subset(dataset, index)
    return (data.DataLoader(balanced_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), lcm)

def print_class_distribution(dataset, output_file=None):
    # print the class distribution of the dataset
    _, counts = np.unique(dataset.targets, return_counts=True)
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

def pseudo_label(model, dataloader, device, K=15, P=5):
    # keep the top P classes for each image
    # for each class; keep the top K images
    # return a new dataloader with num_classes * K images
    # get the predictions
    model.eval()
    shuffle = dataloader.shuffle
    dataloader.shuffle = False
    # for each class, create a list of tuple (index, score)
    lists = [[] for _ in range(model.num_classes)]

    # get the top P classes for each image
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader)):
            # batch has shape (batch_size, 3, H ,W)
            batch.to(device)
            output = model(batch)
            # output has shape (batch_size, num_classes)
            # get the top P classes for each image
            scores, classes = torch.topk(output, P, dim=1)
            # scores has shape (batch_size, P)
            # classes has shape (batch_size, P)
            for i in range(len(batch)):
                for j in range(P):
                    lists[classes[i][j]].append((len(batch) * idx + i, scores[i][j]))
    for i in range(model.num_classes):
        lists[i].sort(key=lambda x: x[1], reverse=True)
        lists[i] = lists[i][:K]
    # lists has shape (num_classes, K)
    # lists[i] is a list of tuple (index, score) for class i
    # create a new dataset
    new_dataset = []
    for i in range(model.num_classes):
        for j in range(K):
            new_dataset.append(dataloader.dataset[lists[i][j][0]], i)
    dataloader.shuffle = shuffle
    return data.DataLoader(new_dataset, batch_size=dataloader.batch_size, shuffle=dataloader.shuffle, num_workers=dataloader.num_workers)

def join(dataloader1, dataloader2):
    # join two dataloaders
    # return a new dataloader
    dataset = data.ConcatDataset([dataloader1.dataset, dataloader2.dataset])
    return data.DataLoader(dataset, batch_size=dataloader1.batch_size, shuffle=dataloader1.shuffle, num_workers=dataloader1.num_workers)



            
            





