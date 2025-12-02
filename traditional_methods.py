from utils import (
    BLUR_KERNEL_SIZE, 
    np, cv2, plt, 
    peak_signal_noise_ratio, structural_similarity,
    load_grayscale, load_blurred_image
)
from visualize import display_image_comparison, comparison_table, plot_image_comparison, plot_metrics_graph
from stochastic_deconvolution import stochastic_deconvolution



## wrapper
def sharpening_technique(technique, original_img, blurred_img):    
    print(f"Applying technique: {technique.__name__}")

    # apply sharpening technique
    sharpened_image, mask = technique(blurred_img)
    
    # display results
    display_image_comparison(blurred_img, sharpened_image, mask)
    
    # compute psnr
    blurred_psnr = peak_signal_noise_ratio(original_img, blurred_img, data_range=1.0)
    sharpened_psnr = peak_signal_noise_ratio(original_img, sharpened_image, data_range=1.0)
    
    return sharpened_psnr, blurred_psnr



## sharpening methods
def unsharp_masking(blurred_img):
    # apply unsharp masking
    gaussian_blur = cv2.GaussianBlur(blurred_img, (7, 7), 0)
    mask = cv2.subtract(blurred_img, gaussian_blur)
    sharpened_image = cv2.addWeighted(blurred_img, 1.0 + 1.5, gaussian_blur, -1.5, 0)
    
    return sharpened_image

def sharpening(blurred_img):
    # create sharpening kernel
    sharpening_kernel = np.array(
        [[0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]]
    )
    
    # apply sharpening filter
    sharpened_image = cv2.filter2D(blurred_img, -1, sharpening_kernel)
    
    return sharpened_image

def laplacian_sharpening(blurred_img):
    blurred_img = blurred_img.astype(np.float32)
    laplacian = cv2.Laplacian(blurred_img, cv2.CV_32F)
    # Subtracting the Laplacian enhances edges
    sharpened = blurred_img - 0.5 * laplacian
    return np.clip(sharpened, 0, 1)

def clahe_enhancement(blurred_img):
    img_uint8 = (blurred_img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_uint8 = clahe.apply(img_uint8)
    return enhanced_uint8.astype(np.float32) / 255.0

def sd(blurred_img, blur_kernel_size=BLUR_KERNEL_SIZE):
    # use stochastic deconvolution to sharpen image
    sharpened_image = stochastic_deconvolution(blurred_img, blur_kernel_size, verbose=False)
    return sharpened_image

def wiener_deconvolution(blurred_img):
    # Wiener filter deconvolution - estimates noise-to-signal ratio
    from scipy.signal import wiener
    # Apply Wiener filter
    sharpened = wiener(blurred_img, (5, 5))
    return np.clip(sharpened, 0, 1)

def bilateral_filter(blurred_img):
    # Bilateral filter - edge-preserving smoothing then sharpening
    img_uint8 = (blurred_img * 255).astype(np.uint8)
    # Apply bilateral filter (preserves edges while reducing noise)
    filtered = cv2.bilateralFilter(img_uint8, 9, 75, 75)
    # Then apply sharpening
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(filtered, -1, sharpening_kernel)
    return sharpened.astype(np.float32) / 255.0

def richardson_lucy(blurred_img, iterations=30):
    # Richardson-Lucy deconvolution algorithm
    from scipy.signal import convolve2d
    # Create a simple Gaussian PSF
    psf = cv2.getGaussianKernel(9, 2.0)
    psf = psf @ psf.T
    psf = psf / psf.sum()
    
    # Richardson-Lucy iterations
    estimate = blurred_img.copy()
    for _ in range(iterations):
        # Convolve estimate with PSF
        conv = convolve2d(estimate, psf, mode='same', boundary='symm')
        # Avoid division by zero
        relative_blur = blurred_img / (conv + 1e-10)
        # Update estimate
        estimate = estimate * convolve2d(relative_blur, psf[::-1, ::-1], mode='same', boundary='symm')
    
    return np.clip(estimate, 0, 1)

def high_pass_filter(blurred_img):
    # High-pass filter to enhance edges
    # Create Gaussian low-pass filter
    low_pass = cv2.GaussianBlur(blurred_img, (21, 21), 5)
    # Subtract from original to get high-pass
    high_pass = blurred_img - low_pass
    # Add back to original with scaling
    sharpened = blurred_img + 1.5 * high_pass
    return np.clip(sharpened, 0, 1)

def median_filter_sharpen(blurred_img):
    # Median filter to reduce noise, then sharpen
    img_uint8 = (blurred_img * 255).astype(np.uint8)
    # Apply median filter
    filtered = cv2.medianBlur(img_uint8, 5)
    # Create strong sharpening kernel
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(filtered, -1, kernel)
    return np.clip(sharpened.astype(np.float32) / 255.0, 0, 1)

def gradient_based_sharpening(blurred_img):
    # Use image gradients to enhance edges
    # Convert to uint8 for Sobel operation
    img_uint8 = (blurred_img * 255).astype(np.uint8)
    # Compute gradients
    grad_x = cv2.Sobel(img_uint8, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_uint8, cv2.CV_32F, 0, 1, ksize=3)
    # Gradient magnitude
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    # Normalize gradient
    gradient_mag = gradient_mag / 255.0
    # Add gradient magnitude to enhance edges
    sharpened = blurred_img + 0.3 * gradient_mag
    return np.clip(sharpened, 0, 1)

## comparison
def run_comparison(methods, filename="dandelion.jpg"):
    original_img = load_grayscale(filename)
    blurred_img = load_blurred_image(filename, blur_kernel_size=BLUR_KERNEL_SIZE)

    base_psnr = peak_signal_noise_ratio(original_img, blurred_img, data_range=1.0)
    base_ssim = structural_similarity(original_img, blurred_img, data_range=1.0)

    results = [{
        "name": "Blurred image",
        "image": blurred_img,
        "psnr": base_psnr,
        "ssim": base_ssim
    }]
    
    for name, func in methods:
        sharpened_img = func(blurred_img)

        psnr = peak_signal_noise_ratio(original_img, sharpened_img, data_range=1.0)
        ssim = structural_similarity(original_img, sharpened_img, data_range=1.0)

        results.append({
            "name": name,
            "image": sharpened_img,
            "psnr": psnr,
            "ssim": ssim
        })
    return results, base_psnr, base_ssim



if __name__ == "__main__":
    methods = [
        ("Unsharp Masking", unsharp_masking),
        ("Sharpening Kernel", sharpening),
        ("Laplacian Sharpening", laplacian_sharpening),
        ("CLAHE Enhancement", clahe_enhancement),
        ("Wiener Deconvolution", wiener_deconvolution),
        ("Bilateral Filter", bilateral_filter),
        ("Richardson-Lucy", richardson_lucy),
        ("High-Pass Filter", high_pass_filter),
        ("Median Filter + Sharpen", median_filter_sharpen),
        ("Gradient-Based Sharpening", gradient_based_sharpening),
        ("Stochastic Deconvolution", sd),
    ]

    results, base_psnr, base_ssim = run_comparison(methods)

    comparison_table(results)
    plot_image_comparison(results[1:])
    plot_metrics_graph(results[1:], base_psnr, base_ssim)

    plt.show()
