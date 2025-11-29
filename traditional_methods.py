from utils import load_grayscale, blur_image_fast

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage


def sharpening():
    # Load a sample image and convert it to grayscale
    # blurred_img = skimage.data.camera()

    try:
        blurred_img = load_grayscale("dandelion.jpg")
        # add extra blurr using psf
        psf = np.zeros((9, 9))
        prepared_PSF_COORDS = []
        for i in range(9):
            px, py = -4 + i, -4 + i
            psf[py + 4, px + 4] = 1.0 / 9.0
            prepared_PSF_COORDS.append((px, py, 1.0 / 9.0))
        # blurr image
        blurred_img = blur_image_fast(blurred_img, psf)
    except FileNotFoundError:
        print("Error: dandelion.jpg not found.")

    #
    gaussian_blur = cv2.GaussianBlur(blurred_img, (7, 7), 0)
    mask = cv2.subtract(blurred_img, gaussian_blur)
    sharpened_image = cv2.addWeighted(blurred_img, 1.0 + 1.5, gaussian_blur, -1.5, 0)

    # display
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(blurred_img, cmap='gray')
    axes[0].set_title('Original Blurry Image')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Edge Mask (Unsharp Mask)')
    axes[1].axis('off')

    axes[2].imshow(sharpened_image, cmap='gray')
    axes[2].set_title('Sharpened via Unsharp Masking')
    axes[2].axis('off')

    plt.show()