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

from scipy.signal import convolve2d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
NUM_EPOCHS = 10
IMAGE_SIZE = 256 # size of cropped images

# other constants
MODEL_SAVE_PATH = "best_model.pth"


def psf(kernel_size=5):
	psf_matrix = np.zeros((kernel_size, kernel_size))

	for i in range(kernel_size):
		px = -2 + i
		py = -2 + i
		psf_matrix[py + (kernel_size // 2), px + (kernel_size // 2)] = 1.0 / kernel_size

	return psf_matrix


def load_grayscale(filename, scale=True):
	"""Loads an image and converts it to a normalized grayscale numpy array."""
	img = Image.open(filename).convert('L')
	converted =	np.array(img, dtype=np.float64)
	if scale:
		converted /= 255.0
	return converted

def blur_image_fast(img, psf):
	"""Blurs an image using the fast SciPy convolution."""
	return convolve2d(img, psf, mode='same', boundary='symm')
