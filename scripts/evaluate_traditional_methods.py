import sys
import os
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import (
    BLUR_KERNEL_SIZE, IMAGE_SIZE,
    np, cv2, plt, 
    peak_signal_noise_ratio, structural_similarity,
)
from src.visualize import comparison_table, plot_image_comparison, plot_metrics_graph
from scripts.stochastic_deconvolution import stochastic_deconvolution
from scripts.richardson_lucy import richardson_lucy
from src.dataset import GoProDataset

NUM_IMAGES_FOR_METRICS = 20


def wiener_deconvolution(blurred_img, mysize=15):
    from scipy.signal import wiener
    
    try:
        if blurred_img.ndim == 3:
            result = np.stack([wiener(blurred_img[:, :, i], (mysize, mysize)) for i in range(3)], axis=2)
            return np.clip(result, 0, 1)
        else:
            result = wiener(blurred_img, (mysize, mysize))
            return np.clip(result, 0, 1)
    except Exception as e:
        print(f"Warning: Wiener deconvolution failed ({e}), returning input")
        return blurred_img

def run_comparison(image_idx=0, save_dir="output/traditional_methods"):
    """Run comparison of traditional deblurring methods."""
    
    test_dataset = GoProDataset(split="test", image_size=IMAGE_SIZE, augment=False)
    
    if image_idx >= len(test_dataset):
        print(f"⚠️  Index {image_idx} out of range (max: {len(test_dataset)-1}). Using 0.")
        image_idx = 0
    
    blurred_tensor, original_tensor = test_dataset[image_idx]
    
    print(f"\n{'='*60}")
    print(f"LOADING DATASET IMAGE")
    print(f"{'='*60}")
    print(f"Image: {test_dataset.blur_images[image_idx]}")
    print(f"Index: {image_idx}")
    
    if original_tensor.shape[0] == 1:
        original_img = original_tensor[0].numpy()
        blurred_img = blurred_tensor[0].numpy()
    else:
        original_img = original_tensor.numpy().transpose(1, 2, 0)
        blurred_img = blurred_tensor.numpy().transpose(1, 2, 0)
    
    def sd_with_original(blurred_img, blur_kernel_size=BLUR_KERNEL_SIZE):
        if blurred_img.ndim == 3:
            blurred_gray = cv2.cvtColor((blurred_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            original_gray = cv2.cvtColor((original_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            sharpened_gray = stochastic_deconvolution(blurred_gray, blur_kernel_size, verbose=False, input_img=original_gray)
            ratio = np.divide(sharpened_gray[:, :, np.newaxis], blurred_gray[:, :, np.newaxis] + 1e-10)
            return np.clip(blurred_img * ratio, 0, 1)
        return stochastic_deconvolution(blurred_img, blur_kernel_size, verbose=False, input_img=original_img)
    
    methods = [
        ("Wiener Deconvolution", wiener_deconvolution),
        ("Richardson-Lucy", richardson_lucy),
        ("Stochastic Deconvolution", sd_with_original),
    ]

    channel_axis = 2 if original_img.ndim == 3 else None
    base_psnr = peak_signal_noise_ratio(original_img, blurred_img, data_range=1.0)
    base_ssim = structural_similarity(original_img, blurred_img, data_range=1.0, channel_axis=channel_axis)

    results = [{"name": "Blurred image", "image": blurred_img, "psnr": base_psnr, "ssim": base_ssim}]
    
    print(f"\n{'='*60}")
    print("PROCESSING METHODS")
    print(f"{'='*60}\n")
    
    for name, func in methods:
        sharpened_img = np.clip(func(blurred_img), 0, 1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            psnr = peak_signal_noise_ratio(original_img, sharpened_img, data_range=1.0)
            if not np.isfinite(psnr):
                psnr = base_psnr
            ssim = structural_similarity(original_img, sharpened_img, data_range=1.0, channel_axis=channel_axis)
            if not np.isfinite(ssim):
                ssim = base_ssim
        
        results.append({"name": name, "image": sharpened_img, "psnr": psnr, "ssim": ssim})
        print(f"✓ {name}")
    
    comparison_table(results)
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("SAVING FIGURES")
    print(f"{'='*60}")
    
    plot_image_comparison(results[1:], save_path=os.path.join(save_dir, "traditional_methods_visual_comparison.png"),
                         original_img=original_img, blurred_img=blurred_img)
    print(f"✓ Visual comparison saved")
    
    plot_metrics_graph(results[1:], base_psnr, base_ssim,
                      save_path=os.path.join(save_dir, "traditional_methods_metrics.png"))
    print(f"✓ Metrics graph saved")
    
    plt.close('all')
    
    return results, base_psnr, base_ssim


def evaluate_metrics_on_multiple_images(num_images=20, save_dir="output/traditional_methods"):
    """Evaluate traditional methods on multiple test images."""
    
    print(f"\n{'='*60}")
    print(f"EVALUATING METRICS ON {num_images} IMAGES")
    print(f"{'='*60}")
    
    test_dataset = GoProDataset(split="test", image_size=IMAGE_SIZE, augment=False)
    num_images = min(num_images, len(test_dataset))
    
    random.seed(42)
    random_indices = random.sample(range(len(test_dataset)), num_images)
    
    def sd_batch(blurred_img, blur_kernel_size=BLUR_KERNEL_SIZE):
        if blurred_img.ndim == 3:
            blurred_gray = cv2.cvtColor((blurred_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            sharpened_gray = stochastic_deconvolution(blurred_gray, blur_kernel_size, verbose=False, input_img=blurred_gray)
            ratio = np.divide(sharpened_gray[:, :, np.newaxis], blurred_gray[:, :, np.newaxis] + 1e-10)
            return np.clip(blurred_img * ratio, 0, 1)
        return stochastic_deconvolution(blurred_img, blur_kernel_size, verbose=False, input_img=blurred_img)
    
    methods = [
        ("Wiener Deconvolution", wiener_deconvolution),
        ("Richardson-Lucy", richardson_lucy),
        ("Stochastic Deconvolution", sd_batch),
    ]
    
    results = {name: {'psnr_sum': 0, 'ssim_sum': 0} for name, _ in methods}
    base_psnr_sum = base_ssim_sum = 0
    
    print(f"\n{'='*60}")
    print("PROCESSING IMAGES")
    print(f"{'='*60}\n")
    
    for idx in random_indices:
        blurred_tensor, original_tensor = test_dataset[idx]
        
        if original_tensor.shape[0] == 1:
            original_img = original_tensor[0].numpy()
            blurred_img = blurred_tensor[0].numpy()
        else:
            original_img = original_tensor.numpy().transpose(1, 2, 0)
            blurred_img = blurred_tensor.numpy().transpose(1, 2, 0)
        
        channel_axis = 2 if original_img.ndim == 3 else None
        base_psnr = peak_signal_noise_ratio(original_img, blurred_img, data_range=1.0)
        base_ssim = structural_similarity(original_img, blurred_img, data_range=1.0, channel_axis=channel_axis)
        
        base_psnr_sum += base_psnr
        base_ssim_sum += base_ssim
        
        for name, func in methods:
            sharpened_img = np.clip(func(blurred_img), 0, 1)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                psnr = peak_signal_noise_ratio(original_img, sharpened_img, data_range=1.0)
                if not np.isfinite(psnr):
                    psnr = base_psnr
                ssim = structural_similarity(original_img, sharpened_img, data_range=1.0, channel_axis=channel_axis)
                if not np.isfinite(ssim):
                    ssim = base_ssim
            
            results[name]['psnr_sum'] += psnr
            results[name]['ssim_sum'] += ssim
        
        processed = random_indices.index(idx) + 1
        if processed % 5 == 0:
            print(f"✓ Processed {processed}/{num_images} images")
    
    avg_base_psnr = base_psnr_sum / num_images
    avg_base_ssim = base_ssim_sum / num_images
    
    print(f"\n{'='*80}")
    print(f"METRICS EVALUATION RESULTS ({num_images} images)")
    print(f"{'='*80}")
    print(f"{'Method':<25} | {'Avg PSNR':<12} | {'Avg SSIM':<12} | {'PSNR Improvement':<15}")
    print("-"*80)
    
    for name, _ in methods:
        avg_psnr = results[name]['psnr_sum'] / num_images
        avg_ssim = results[name]['ssim_sum'] / num_images
        improvement = avg_psnr - avg_base_psnr
        print(f"{name:<25} | {avg_psnr:>10.2f} dB | {avg_ssim:>10.4f} | {improvement:>13.2f} dB")
    
    print("="*80)
    print(f"Blurred Input Baseline: PSNR={avg_base_psnr:.2f} dB, SSIM={avg_base_ssim:.4f}")
    print("="*80)
    
    best_method = max(methods, key=lambda x: results[x[0]]['psnr_sum'])
    best_psnr = results[best_method[0]]['psnr_sum'] / num_images
    print(f"\n🏆 Best: {best_method[0]} ({best_psnr:.2f} dB)")
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("SAVING METRICS CHART")
    print(f"{'='*60}")
    
    method_names = [name for name, _ in methods]
    avg_psnrs = [results[name]['psnr_sum'] / num_images for name, _ in methods]
    avg_ssims = [results[name]['ssim_sum'] / num_images for name, _ in methods]
    x = range(len(method_names))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Traditional Methods - Performance Metrics Comparison', fontsize=16)

    psnr_bars = ax1.bar(x, avg_psnrs, color='skyblue', label='Avg PSNR')
    ax1.axhline(y=avg_base_psnr, color='r', linestyle='--', label=f'Blurred Input ({avg_base_psnr:.2f})')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Peak Signal-to-Noise Ratio')
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_names, rotation=15, ha="right")
    ax1.legend()
    ax1.bar_label(psnr_bars, padding=3, fmt='%.2f')

    ssim_bars = ax2.bar(x, avg_ssims, color='salmon', label='Avg SSIM')
    ax2.axhline(y=avg_base_ssim, color='r', linestyle='--', label=f'Blurred Input ({avg_base_ssim:.4f})')
    ax2.set_ylabel('SSIM')
    ax2.set_title('Structural Similarity Index')
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_names, rotation=15, ha="right")
    ax2.legend()
    ax2.bar_label(ssim_bars, padding=3, fmt='%.4f')
    ax2.set_ylim([min(avg_base_ssim, min(avg_ssims)) - 0.05, 1.0])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, 'traditional_methods_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Metrics chart saved")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare traditional deblurring methods')
    parser.add_argument('--index', type=int, default=None, help='Dataset image index for single image visual comparison')
    parser.add_argument('--save-dir', type=str, default='output/traditional_methods', help='Output directory (default: output/traditional_methods/)')
    parser.add_argument('--num-images', type=int, default=NUM_IMAGES_FOR_METRICS, 
                        help=f'Number of images for metrics evaluation (default: {NUM_IMAGES_FOR_METRICS})')
    args = parser.parse_args()
    
    if args.index is not None:
        run_comparison(image_idx=args.index, save_dir=args.save_dir)
    else:
        evaluate_metrics_on_multiple_images(num_images=args.num_images, save_dir=args.save_dir)

