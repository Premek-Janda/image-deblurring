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

import cv2
from scipy.signal import convolve2d
from skimage.metrics import peak_signal_noise_ratio

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
NUM_EPOCHS = 10
IMAGE_SIZE = 256 # size of cropped images

# other constants
MODEL_SAVE_PATH = "best_model.pth"


def blur_psf(kernel_size=9):
	"""Creates a normalized blur kernel (point spread function) with a given kernel size."""
	psf_matrix = np.zeros((kernel_size, kernel_size))

	for i in range(kernel_size):
		psf_matrix[i, i] = 1.0 / kernel_size

	return psf_matrix

def blur_psf_coords(kernel_size=9):
	"""Creates a list of PSF coordinates and their values for convolution."""
	psf_coords = []
	for i in range(kernel_size):
		pxy = i - (kernel_size // 2)
		psf_coords.append((pxy, pxy, 1.0 / kernel_size))
	return np.array(psf_coords)


def load_grayscale(filename, scale=True):
	"""Loads an image and converts it to a normalized grayscale numpy array."""
	try:
		img = Image.open(filename).convert('L')
		converted =	np.array(img, dtype=np.float64)
		if scale:
			converted /= 255.0
		return converted
	except FileNotFoundError:
		print(f"Error: {filename} not found.")

def blur_image_fast(img, psf):
	"""Blurs an image using the fast SciPy convolution."""
	return convolve2d(img, psf, mode='same', boundary='symm')

def load_blurred_image(filename, blur_kernel=5, scale=True):
	"""Loads an image, converts it to grayscale, and applies blur using the given PSF."""
	img = load_grayscale(filename, scale=scale)
	psf = blur_psf(kernel_size=blur_kernel)
	blurred_img = blur_image_fast(img, psf=psf)
	return blurred_img
