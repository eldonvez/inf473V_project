from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import imp
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim 
import torchnet as tnt