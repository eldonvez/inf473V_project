import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import wandb
import hydra

@hydra.main(config_path="configs", config_name="config")
def train(cfg):
    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    criterion = hydra.utils.instantiate(cfg.criterion)
    train_loader = hydra.utils.instantiate(cfg.train_loader)
    test_loader = hydra.utils.instantiate(cfg.test_loader)
    device = "gpu" if torch.cuda.is_available() else "cpu"



if __name__ == "__main__":
    train()