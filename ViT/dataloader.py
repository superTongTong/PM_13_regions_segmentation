from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

import torchvision.transforms.functional as TF
from torch.autograd import Variable
import torch.nn.functional as F
import io
import nibabel  # to read .hdr/.img files
import numpy
from vit_yao import vit_abdomen
#from vit_pytorch.efficient import ViT
#from vit_pytorch import ViT
# from vit_pytorch import ViT3

def resize2d(img, size):
    return F.adaptive_avg_pool2d(Variable(img),size).data

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # Training settings
    batch_size = 4  # 64
    epochs = 40  # 10   #20 #10 #50 #20
    lr = 3e-5
    gamma = 0.7
    seed = 42  # 42
    seed_everything(seed)

def load_data_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall()