import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.restoration import richardson_lucy as rl_deconv


def detect_smooth_regions(img, window_size=15, threshold=0.02):
    from skimage import color
    
    if img.ndim == 3:
        gray = color.rgb2gray(img)
    else:
        gray = img
    
    local_std = ndimage.generic_filter(gray, np.std, size=window_size)
    smooth_mask = local_std < threshold
    return smooth_mask


def create_edge_mask(img, canny_low=30, canny_high=100, dilate_size=7):
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    
    edges = cv2.Canny(img_uint8, canny_low, canny_high)
    kernel = np.ones((dilate_size, dilate_size), np.uint8)
    edge_mask = cv2.dilate(edges, kernel, iterations=2)
    
    return edge_mask / 255.0


def richardson_lucy(blurred_img, iterations=10, sigma=0.0, use_wrap=True, suppress_ringing=True):
    try:
        psf_size = 11
        psf = np.zeros((psf_size, psf_size))
        psf[psf_size // 2, :] = 1
        
        psf = gaussian_filter(psf, sigma=sigma)
        psf = psf / psf.sum()
        
        half_iter = max(iterations // 2, 5)
        
        if blurred_img.ndim == 3:
            initial_result = np.zeros_like(blurred_img)
            for c in range(3):
                if use_wrap:
                    pad_size = psf_size * 2
                    channel_padded = np.pad(blurred_img[:, :, c], pad_size, mode='wrap')
                    deconv_padded = rl_deconv(channel_padded, psf, num_iter=half_iter, clip=False)
                    initial_result[:, :, c] = deconv_padded[pad_size:-pad_size, pad_size:-pad_size]
                else:
                    initial_result[:, :, c] = rl_deconv(blurred_img[:, :, c], psf, num_iter=half_iter, clip=False)
            initial_result = np.clip(initial_result, 0, 1)
        else:
            if use_wrap:
                pad_size = psf_size * 2
                img_padded = np.pad(blurred_img, pad_size, mode='wrap')
                deconv_padded = rl_deconv(img_padded, psf, num_iter=half_iter, clip=False)
                initial_result = deconv_padded[pad_size:-pad_size, pad_size:-pad_size]
            else:
                initial_result = rl_deconv(blurred_img, psf, num_iter=half_iter, clip=False)
            initial_result = np.clip(initial_result, 0, 1)
        
        if not suppress_ringing:
            if blurred_img.ndim == 3:
                result = np.zeros_like(blurred_img)
                for c in range(3):
                    if use_wrap:
                        pad_size = psf_size * 2
                        channel_padded = np.pad(blurred_img[:, :, c], pad_size, mode='wrap')
                        deconv_padded = rl_deconv(channel_padded, psf, num_iter=iterations, clip=False)
                        result[:, :, c] = deconv_padded[pad_size:-pad_size, pad_size:-pad_size]
                    else:
                        result[:, :, c] = rl_deconv(blurred_img[:, :, c], psf, num_iter=iterations, clip=False)
                return np.clip(result, 0, 1)
            else:
                if use_wrap:
                    pad_size = psf_size * 2
                    img_padded = np.pad(blurred_img, pad_size, mode='wrap')
                    deconv_padded = rl_deconv(img_padded, psf, num_iter=iterations, clip=False)
                    result = deconv_padded[pad_size:-pad_size, pad_size:-pad_size]
                else:
                    result = rl_deconv(blurred_img, psf, num_iter=iterations, clip=False)
                return np.clip(result, 0, 1)
        
        edge_mask = create_edge_mask(initial_result)
        smooth_mask = detect_smooth_regions(initial_result)
        
        if blurred_img.ndim == 3:
            second_pass = np.zeros_like(blurred_img)
            for c in range(3):
                if use_wrap:
                    pad_size = psf_size * 2
                    channel_padded = np.pad(blurred_img[:, :, c], pad_size, mode='wrap')
                    deconv_padded = rl_deconv(channel_padded, psf, num_iter=iterations, clip=False)
                    second_pass[:, :, c] = deconv_padded[pad_size:-pad_size, pad_size:-pad_size]
                else:
                    second_pass[:, :, c] = rl_deconv(blurred_img[:, :, c], psf, num_iter=iterations, clip=False)
            second_pass = np.clip(second_pass, 0, 1)
        else:
            if use_wrap:
                pad_size = psf_size * 2
                img_padded = np.pad(blurred_img, pad_size, mode='wrap')
                deconv_padded = rl_deconv(img_padded, psf, num_iter=iterations, clip=False)
                second_pass = deconv_padded[pad_size:-pad_size, pad_size:-pad_size]
            else:
                second_pass = rl_deconv(blurred_img, psf, num_iter=iterations, clip=False)
            second_pass = np.clip(second_pass, 0, 1)
        
        gradient_strength = 0.3
        smooth_regions = smooth_mask & (edge_mask < 0.3)
        
        result = second_pass.copy()
        
        if blurred_img.ndim == 3:
            for c in range(3):
                channel = second_pass[:, :, c]
                filtered = ndimage.median_filter(channel, size=3)
                result[:, :, c][smooth_regions] = (1 - gradient_strength) * channel[smooth_regions] + \
                                                    gradient_strength * filtered[smooth_regions]
                result[:, :, c][edge_mask > 0.5] = initial_result[:, :, c][edge_mask > 0.5]
        else:
            filtered = ndimage.median_filter(second_pass, size=3)
            result[smooth_regions] = (1 - gradient_strength) * second_pass[smooth_regions] + \
                                      gradient_strength * filtered[smooth_regions]
            result[edge_mask > 0.5] = initial_result[edge_mask > 0.5]
        
        return np.clip(result, 0, 1)

    except Exception as e:
        print(f"Warning: Richardson-Lucy advanced failed ({e}), returning input")
        return blurred_img
