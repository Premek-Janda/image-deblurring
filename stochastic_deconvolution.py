from utils import (
    BLUR_KERNEL_SIZE,
    peak_signal_noise_ratio,
    blur_psf, blur_psf_coords,
    load_grayscale, blur_image_fast, 
)

import numpy as np
from PIL import Image
import os
import math
import numba

# output
SAVE_DIR = "output"

# method Parameters
REG_WEIGHT = 0.0005
SIGMA = 4.0
RESET_PROB = 0.005
NUM_ITERATIONS = 100
ED_START = 0.025

REG_STENCIL = np.array([[0, 1, 0], [0, 0, 1]]) # [reg_x], [reg_y]

# core functions
@numba.jit(nopython=True)
def data_energy(blurred_img, input_img, x, y, psf_coords):
    """Computes the data term energy for a pixel."""
    height, width = blurred_img.shape
    sum_sq_diff = 0.0
    for i in range(len(psf_coords)):
        px, py, pv = psf_coords[i]
        tx, ty = x + int(px), y + int(py)
        if 0 <= tx < width and 0 <= ty < height:
            delta = blurred_img[ty, tx] - input_img[ty, tx]
            sum_sq_diff += delta * delta
    return sum_sq_diff

@numba.jit(nopython=True)
def regularizer_energy(intrinsic_img, x, y):
    """Evaluates the isotropic TV regularizer energy at a pixel."""
    height, width = intrinsic_img.shape
    if not (0 <= x < width and 0 <= y < height):
        return 0.0
    dx = intrinsic_img[y, x] - intrinsic_img[y, x - 1] if x > 0 else 0.0
    dy = intrinsic_img[y, x] - intrinsic_img[y - 1, x] if y > 0 else 0.0
    return REG_WEIGHT * math.sqrt(dx * dx + dy * dy)

@numba.jit(nopython=True)
def splat(intrinsic_img, blurred_img, x, y, ed, weight, psf_coords):
    """Splats energy into the intrinsic and blurred images."""
    height, width = blurred_img.shape
    intrinsic_img[y, x] += weight * ed
    for i in range(len(psf_coords)):
        px, py, pv = psf_coords[i]
        tx, ty = x + int(px), y + int(py)
        if 0 < tx < width and 0 < ty < height:
            blurred_img[ty, tx] += weight * ed * pv

@numba.jit(nopython=True)
def evaluate(intrinsic_img, blurred_img, input_img, x, y, ed, psf_coords):
    """Evaluates the change in the objective function."""
    init_energy = data_energy(blurred_img, input_img, x, y, psf_coords)
    for i in range(REG_STENCIL.shape[1]):
        init_energy += regularizer_energy(intrinsic_img, x + REG_STENCIL[0, i], y + REG_STENCIL[1, i])

    splat(intrinsic_img, blurred_img, x, y, ed, 1.0, psf_coords)
    plus_energy = data_energy(blurred_img, input_img, x, y, psf_coords)
    for i in range(REG_STENCIL.shape[1]):
        plus_energy += regularizer_energy(intrinsic_img, x + REG_STENCIL[0, i], y + REG_STENCIL[1, i])

    splat(intrinsic_img, blurred_img, x, y, ed, -2.0, psf_coords)
    minus_energy = data_energy(blurred_img, input_img, x, y, psf_coords)
    for i in range(REG_STENCIL.shape[1]):
        minus_energy += regularizer_energy(intrinsic_img, x + REG_STENCIL[0, i], y + REG_STENCIL[1, i])

    splat(intrinsic_img, blurred_img, x, y, ed, 1.0, psf_coords)

    de_plus = init_energy - plus_energy
    de_minus = init_energy - minus_energy

    ed_sign = -1.0 if de_minus > de_plus else 1.0
    max_de = max(de_plus, de_minus)
    return max_de, ed_sign

@numba.jit(nopython=True)
def mutate(intrinsic_shape, cur_x, cur_y):
    """Mutates a sample by offsetting or resetting."""
    height, width = intrinsic_shape
    if np.random.rand() >= RESET_PROB:
        while True:
            dx, dy = np.random.normal(0, 1, 2)
            new_x = int(cur_x + SIGMA * dx + 0.5)
            new_y = int(cur_y + SIGMA * dy + 0.5)
            if (new_x != cur_x or new_y != cur_y) and (0 <= new_x < width and 0 <= new_y < height):
                return new_x, new_y
    else:
        return np.random.randint(0, width), np.random.randint(0, height)

@numba.jit(nopython=True)
def stochastic_deconvolution_core(intrinsic_img, blurred_img, input_img, ed, n_mutations, psf_coords):
    """The core deconvolution loop, accelerated with Numba."""
    height, width = input_img.shape
    a_rate = 0.0

    # Initialize first sample
    sample_x_x, sample_x_y = np.random.randint(0, width), np.random.randint(0, height)
    sample_x_ed = ed

    fx, ed_sign = evaluate(intrinsic_img, blurred_img, input_img, sample_x_x, sample_x_y, sample_x_ed, psf_coords)
    sample_x_ed *= ed_sign

    for _ in range(n_mutations):
        sample_y_x, sample_y_y = mutate(intrinsic_img.shape, sample_x_x, sample_x_y)
        # Start with the same energy magnitude
        sample_y_ed = sample_x_ed 

        fy, ed_sign = evaluate(intrinsic_img, blurred_img, input_img, sample_y_x, sample_y_y, sample_y_ed, psf_coords)
        sample_y_ed *= ed_sign

        if fy > 0.0:
            a_rate += 1.0
            splat(intrinsic_img, blurred_img, sample_y_x, sample_y_y, sample_y_ed, 1.0, psf_coords)

        accept_prob = 0.0
        if fx != 0:
            accept_prob = min(1.0, fy / fx)

        if (fx <= 0.0 and fy >= fx) or (np.random.rand() < accept_prob):
            sample_x_x, sample_x_y, sample_x_ed = sample_y_x, sample_y_y, sample_y_ed
            fx, ed_sign = evaluate(intrinsic_img, blurred_img, input_img, sample_x_x, sample_x_y, sample_x_ed, psf_coords)
            sample_x_ed *= ed_sign
            
    return a_rate / n_mutations


def stochastic_deconvolution(blurred_img=None, blur_kernel_size=BLUR_KERNEL_SIZE, verbose=True):
    """Wrapper for the Numba-accelerated stochastic deconvolution core."""
    # Create output directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    input_img = load_grayscale("dandelion.jpg")
    
    psf = blur_psf(blur_kernel_size)
    psf_coords = blur_psf_coords(blur_kernel_size)
    
    if blurred_img is None:
        blurred_img = blur_image_fast(input_img, psf)
    
    # create copies for the algorithm to modify
    intrinsic_img = blurred_img.copy()
    current_blurred = blur_image_fast(intrinsic_img, psf)
    
    # "warm-up" run
    _ = stochastic_deconvolution_core(np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), 0.1, 1, psf_coords)

    ed = ED_START
    for k in range(NUM_ITERATIONS):
        accept_rate = stochastic_deconvolution_core(intrinsic_img, current_blurred, blurred_img, ed, input_img.size, psf_coords)
        if verbose:
            print(f"Iteration {k+1}/{NUM_ITERATIONS}, Acceptance Rate: {accept_rate:.4f}, ed: {ed:.4f}")
        if accept_rate < 0.4:
            ed *= 0.5

    Image.fromarray((np.clip(input_img, 0, 1) * 255).astype(np.uint8)).save(os.path.join(SAVE_DIR, "ground_truth.png"))
    Image.fromarray((np.clip(blurred_img, 0, 1) * 255).astype(np.uint8)).save(os.path.join(SAVE_DIR, "blurred.png"))
    Image.fromarray((np.clip(intrinsic_img, 0, 1) * 255).astype(np.uint8)).save(os.path.join(SAVE_DIR, "intrinsic.png"))
    
    blurred_psnr = peak_signal_noise_ratio(input_img, blurred_img, data_range=1.0)
    intrinsic_psnr = peak_signal_noise_ratio(input_img, intrinsic_img, data_range=1.0)
    
    if verbose:
        print(f"PSNR of blurred image: {blurred_psnr:.2f} dB")
        print(f"Final PSNR of deconvolved image: {intrinsic_psnr:.2f} dB")
    
    return intrinsic_img
    


if __name__ == '__main__':
    stochastic_deconvolution()