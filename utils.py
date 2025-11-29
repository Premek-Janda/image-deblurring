### constants and utility functions 
import os
import random

import numpy as np

import kagglehub

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset, DataLoader
from torchmetrics.image import PeakSignalNoiseRatio

from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
NUM_EPOCHS = 10
IMAGE_SIZE = 256 # size of cropped images

# other constants
MODEL_SAVE_PATH = "best_model.pth"
