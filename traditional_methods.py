from utils import (
    np, cv2, plt, 
    peak_signal_noise_ratio,
    load_grayscale, load_blurred_image
)
from visualize import display_image_comparison
from stochastic_deconvolution import stochastic_deconvolution

BLUR_KERNEL_SIZE = 5

def sharpening_technique(technique):
    original_img = load_grayscale("dandelion.jpg")
    blurred_img = load_blurred_image("dandelion.jpg", blur_kernel=BLUR_KERNEL_SIZE)

    # apply sharpening technique
    sharpened_image, mask = technique(blurred_img)
    
    # display results
    display_image_comparison(blurred_img, sharpened_image, mask)
    
    # compute psnr
    blurred_psnr = peak_signal_noise_ratio(original_img, blurred_img, data_range=1.0)
    sharpened_psnr = peak_signal_noise_ratio(original_img, sharpened_image, data_range=1.0)
    
    return sharpened_psnr, blurred_psnr

def unsharp_masking(blurred_img):
    # apply unsharp masking
    gaussian_blur = cv2.GaussianBlur(blurred_img, (7, 7), 0)
    mask = cv2.subtract(blurred_img, gaussian_blur)
    sharpened_image = cv2.addWeighted(blurred_img, 1.0 + 1.5, gaussian_blur, -1.5, 0)
    
    return sharpened_image, mask

def sharpening(blurred_img):
    # create sharpening kernel
    sharpening_kernel = np.array(
        [[0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]]
    )
    
    # apply sharpening filter
    sharpened_image = cv2.filter2D(blurred_img, -1, sharpening_kernel)
    
    return sharpened_image, None

def sd(blurred_img):
    # use stochastic deconvolution to sharpen image
    sharpened_image = stochastic_deconvolution(blurred_img=blurred_img, blur_kernel_size=BLUR_KERNEL_SIZE, verbose=False)
    
    return sharpened_image, None

if __name__ == "__main__":
    res_unsharp_masking, blurred = sharpening_technique(unsharp_masking)
    res_sharpening, _ = sharpening_technique(sharpening)
    res_sd, _ = sharpening_technique(sd)
    
    
    print("Displaying Unsharp Masking Results")
    print(f"Blurred psnr:             {blurred:.2f} dB")
    print(f"Unsharp masking:          {res_unsharp_masking:.2f} dB")
    print(f"Sharpening:               {res_sharpening:.2f} dB")
    print(f"Stochastic deconvolution: {res_sd:.2f} dB")
    
    plt.show()
    
