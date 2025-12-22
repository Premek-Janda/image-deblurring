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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 1e-4
BATCH_SIZE = 2
NUM_EPOCHS = 10
IMAGE_SIZE = 256
BLUR_KERNEL_SIZE = 5
MODEL_SAVE_PATH = "best_model.pth"

def blur_psf(kernel_size=BLUR_KERNEL_SIZE):
	psf_matrix = np.zeros((kernel_size, kernel_size))
	for i in range(kernel_size):
		psf_matrix[i, i] = 1.0 / kernel_size
	return psf_matrix

def blur_psf_coords(kernel_size=BLUR_KERNEL_SIZE):
	psf_coords = []
	for i in range(kernel_size):
		pxy = i - (kernel_size // 2)
		psf_coords.append((pxy, pxy, 1.0 / kernel_size))
	return np.array(psf_coords)

def load_grayscale(filename, scale=True):
	try:
		img = Image.open(filename).convert('L')
		converted =	np.array(img, dtype=np.float64)
		if scale:
			converted /= 255.0
		return converted
	except FileNotFoundError:
		print(f"Error: {filename} not found.")

def blur_image_fast(img, psf):
	return convolve2d(img, psf, mode='same', boundary='symm')

def load_blurred_image(filename, blur_kernel_size=BLUR_KERNEL_SIZE, scale=True):
	img = load_grayscale(filename, scale=scale)
	psf = blur_psf(kernel_size=blur_kernel_size)
	blurred_img = blur_image_fast(img, psf=psf)
	return blurred_img
