import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import utils
import wandb
import hydra
import itertools
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = wandb.init(project="inf473v", name="run")

@hydra.main(config_path="configs", config_name="config")
def train(cfg):
    teacher = hydra.utils.instantiate(cfg.teacher)
    optimizer = hydra.utils.instantiate(cfg.optimizer, teacher.parameters())
    criterion = hydra.utils.instantiate(cfg.criterion)
    train_set = hydra.utils.instantiate(cfg.train_set)
    unlabeled_dataset = hydra.utils.instantiate(cfg.unlabeled_dataset)
    teacher.to(device)
    wandb.watch(teacher)
    cross_folds = utils.generate_folds(cfg.data_dir, cfg.num_folds)
    val_acc = [0]*cfg.num_folds

    for i, (train_loader, val_loader) in enumerate(cross_folds):
        for epoch in tqdm(range(cfg.warmup_epochs)):
            epoch_loss = 0
            epoch_num_correct = 0
            num_samples = 0
            for j, batch in enumerate(train_loader):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                preds = teacher(images)
                loss = criterion(preds, labels)
                logger.log({"loss": loss.detach().cpu().numpy()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().numpy() * len(images)
                epoch_num_correct += (
                    (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                )
                num_samples += len(images)
            epoch_loss /= num_samples
            epoch_acc = epoch_num_correct / num_samples
            logger.log(
                {
                    "epoch": epoch,
                    "train_loss_epoch": epoch_loss,
                    "train_acc": epoch_acc,
                }
            )
            epoch_loss = 0
            epoch_num_correct = 0
            num_samples = 0

            for j, batch in enumerate(val_loader):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                preds = teacher(images)
                loss = criterion(preds, labels)
                epoch_loss += loss.detach().cpu().numpy() * len(images)
                epoch_num_correct += (
                    (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                )
                num_samples += len(images)
            epoch_loss /= num_samples
            epoch_acc = epoch_num_correct / num_samples
            logger.log(
                {
                    "epoch": epoch,
                    "val_loss_epoch": epoch_loss,
                    "val_acc": epoch_acc,
                }
            )
        torch.save(teacher.state_dict(), cfg.teacher_path + str(i) +"warmup"+".pth")
        pseudo_labels = None
        #ensures gradual ramp up in relative weight of pseudo label loss in such a way that the average weight is cfg.weight
        max_weight = 2 * (cfg.warmup_epochs + cfg.epochs) / cfg.epochs * cfg.weight 
        for epoch in tqdm(range(cfg.epochs)):
            pseudo_labels, unlabeled_dataset = utils.generate_pseudo_labels(teacher, unlabeled_dataset, pseudo_labels, device, cfg.batch_size)
            teacher.train()
            epoch_loss = 0
            epoch_num_correct = 0
            num_samples = 0
            weight = epoch / (cfg.epochs) * max_weight
            # feed one batch at a time of labeled and unlabeled data
            pseudo_loader = DataLoader(unlabeled_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
            loader = itertools.zip_longest(train_loader, pseudo_loader)
            for i, (batch, pseudo_batch) in enumerate(loader):
                if batch is not None:
                    batch.to(device)
                    images, labels = batch
                    preds = teacher(images)
                    loss = criterion(preds, labels)
                    logger.log({"loss": loss.detach().cpu().numpy()})
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().cpu().numpy() * len(images)
                    epoch_num_correct += (
                        (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                    )
                    num_samples += len(images)
                if pseudo_batch is not None:
                    images, labels = pseudo_batch
                    images = images.to(device)
                    labels = labels.to(device)
                    preds = teacher(images)
                    loss = criterion(preds, labels)
                    loss = weight * loss
                    logger.log({"loss": loss.detach().cpu().numpy()})
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().cpu().numpy() * len(images)
                    epoch_num_correct += (
                        (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                    )
                    num_samples += len(images)

            epoch_loss /= num_samples
            epoch_acc = epoch_num_correct / num_samples
            logger.log(
                {
                    "epoch": epoch,
                    "train_loss_epoch": epoch_loss,
                    "train_acc": epoch_acc,
                }
            )
            epoch_loss = 0
            epoch_num_correct = 0
            num_samples = 0
            teacher.eval()
            for j, batch in enumerate(val_loader):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                preds = teacher(images)
                loss = criterion(preds, labels)
                epoch_loss += loss.detach().cpu().numpy() * len(images)
                epoch_num_correct += (
                    (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                )
                num_samples += len(images)
            epoch_loss /= num_samples
            epoch_acc = epoch_num_correct / num_samples
            logger.log(
                {
                    "epoch": epoch,
                    "val_loss_epoch": epoch_loss,
                    "val_acc": epoch_acc,
                }
            )
        torch.save(teacher.state_dict(), cfg.teacher_path + str(i) + "pseudo" + ".pth")
        #reset unlabeled dataset
        unlabeled_dataset = hydra.utils.instantiate(cfg.unlabeled_dataset)
        # reset teacher
        teacher = hydra.utils.instantiate(cfg.teacher)
        teacher.to(device)
        pseudo_labels = None

    val_acc = np.array(val_acc)
    print("Average val accuracy: ", np.mean(val_acc))
    print("Standard deviation: ", np.std(val_acc))



       



    


    
        



if __name__ == "__main__":
    train()