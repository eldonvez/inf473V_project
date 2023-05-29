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



def generate_pseudo_labels(model, unlabeled_dataset, pseudo_labeled_dataset, device, batch_size=32, K=15, P=5):
    # keep the top P classes for each image
    # for each class; keep the top K images
    # return a new dataloader with num_classes * K images
    # get the predictions
    model.eval()
    dataloader = data.DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False)
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
    # create unlabeled_dataset
    new_dataset = []
    indices = []
    for i in range(model.num_classes):
        for j in range(K):
            index = lists[i][j][0]
            new_dataset.append(dataloader.dataset[index], i)
            indices.append(index)
    # turn new_dataset into a Dataset object
    new_dataset = data.TensorDataset(torch.stack([x[0] for x in new_dataset]), torch.tensor([x[1] for x in new_dataset]))
    # remove the pseudo labels from the original dataset
    new_unlabeled = data.Subset(dataloader.dataset, list(set(range(len(dataloader.dataset))) - set(indices)))
    if pseudo_labeled_dataset is None:
        pseudo_labeled_dataset = new_dataset
    else:   
        pseudo_labeled_dataset = data.ConcatDataset([pseudo_labeled_dataset, new_dataset])
    return (pseudo_labeled_dataset, new_unlabeled)

def get_run_name(cfg):
    run_name = (f"{cfg.teacher.name}_{'frozen' if cfg.teacher.frozen else 'unfrozen'}_"
        f"{'selfTrain' if cfg.self_train else (cfg.student.name+('_pretrained' if cfg.student.pretrained else ''))}_"
        f"{cfg.dataset.transform.name}_{cfg.dataset.name}_{cfg.warmup_epochs}warmup_"
        f"{cfg.epochs}epochs_{cfg.datamodule.batch_size}batch_{cfg.optimizer._target_}_"
        f"{cfg.optimizer.lr}lr_{'Xvalid' if cfg.datamodule.cross_validation else ''}_"
        f"{cfg.max_weight}pseudoloss")
    return (run_name)
            
            





