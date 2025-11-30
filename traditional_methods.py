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
        ("Stochastic Deconvolution", sd),
    ]

    results, base_psnr, base_ssim = run_comparison(methods)

    comparison_table(results)
    plot_image_comparison(results[1:])
    plot_metrics_graph(results[1:], base_psnr, base_ssim)

    plt.show()
