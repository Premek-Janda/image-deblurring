from src.utils import (
    DEVICE, BLUR_KERNEL_SIZE,
    os, np, plt,
    nn, torch, PeakSignalNoiseRatio,
    load_blurred_image, load_grayscale
)
from src.models import DeblurringSimple, UNet, DeblurCNN

def plot_one_image_comparison(model, loader, eval_metric, title="", save_path=None):
    from skimage.metrics import structural_similarity
    
    model.eval()
    data_blur, data_sharp = next(iter(loader))
    data_blur = data_blur.to(DEVICE)
    data_sharp = data_sharp.to(DEVICE)
    
    with torch.no_grad():
        deblurred = model(data_blur)
        
    # input vs target - PSNR
    eval_input = eval_metric(data_blur[0].unsqueeze(0), data_sharp[0].unsqueeze(0)).item()
    # output vs target - PSNR
    eval_output = eval_metric(deblurred[0].unsqueeze(0), data_sharp[0].unsqueeze(0)).item()
    
    def prepare_img(tensor):
        img = tensor[0].cpu().permute(1, 2, 0).numpy()
        return np.clip(img, 0, 1)

    blur_img = prepare_img(data_blur)
    deblur_img = prepare_img(deblurred)
    sharp_img = prepare_img(data_sharp)
    
    # Calculate SSIM
    ssim_input = structural_similarity(blur_img.squeeze() if blur_img.ndim == 3 and blur_img.shape[2] == 1 else blur_img, 
                                       sharp_img.squeeze() if sharp_img.ndim == 3 and sharp_img.shape[2] == 1 else sharp_img, 
                                       data_range=1.0, channel_axis=2 if blur_img.ndim == 3 and blur_img.shape[2] > 1 else None)
    ssim_output = structural_similarity(deblur_img.squeeze() if deblur_img.ndim == 3 and deblur_img.shape[2] == 1 else deblur_img, 
                                        sharp_img.squeeze() if sharp_img.ndim == 3 and sharp_img.shape[2] == 1 else sharp_img, 
                                        data_range=1.0, channel_axis=2 if deblur_img.ndim == 3 and deblur_img.shape[2] > 1 else None)


    plt.figure(figsize=(12, 4))
    plt.suptitle(title)
    
    plt.subplot(1, 3, 1)
    plt.imshow(blur_img)
    plt.title(f"Blurred Input - PSNR: {eval_input:.2f}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(deblur_img)
    plt.title(f"Model Output - PSNR: {eval_output:.2f}")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(sharp_img)
    plt.title("Target")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    
    return eval_input, eval_output, ssim_input, ssim_output

def plot_kernel_comparison(model):
    if hasattr(model, 'convolution'):
        weights = model.convolution.weight.data[0, 0].cpu().numpy()
        plt.figure(figsize=(4, 4))
        plt.imshow(weights, cmap='viridis')
        plt.colorbar()
        plt.title("Learned Kernel (Channel 0)")
    else:
        print("Model does not have a single convolution layer to inspect.")
def comparison_table(results):
    print(f"{'Method':<25} | {'PSNR (dB)':<12} | {'SSIM':<12}")
    for res in results:
        print(f"{res['name']:<25} | {res['psnr']:<12.2f} | {res['ssim']:<12.4f}")

def plot_image_comparison(results, blur_kernel_size=BLUR_KERNEL_SIZE, save_path=None, original_img=None, blurred_img=None):
    if original_img is None or blurred_img is None:
        raise ValueError("original_img and blurred_img must be provided")

    num_images = len(results) + 2
    cols = 3
    rows = (num_images + cols) // cols
    plt.figure(figsize=(12, 3 * rows))
    plt.suptitle("Visual Comparison of Deblurring Methods", fontsize=16)

    # Determine if images are grayscale or RGB
    is_grayscale = original_img.ndim == 2 or (original_img.ndim == 3 and original_img.shape[2] == 1)
    cmap = 'gray' if is_grayscale else None

    plt.subplot(rows, cols, 1)
    plt.imshow(original_img, cmap=cmap)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(rows, cols, 2)
    plt.imshow(blurred_img, cmap=cmap)
    plt.title("Blurred")
    plt.axis('off')

    for i, res in enumerate(results):
        plt.subplot(rows, cols, i + 4)
        # Check if result image is grayscale or RGB
        res_is_grayscale = res['image'].ndim == 2 or (res['image'].ndim == 3 and res['image'].shape[2] == 1)
        res_cmap = 'gray' if res_is_grayscale else None
        plt.imshow(res['image'], cmap=res_cmap)
        plt.title(f"{res['name']}\nPSNR: {res['psnr']:.2f}, SSIM: {res['ssim']:.4f}")
        plt.axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visual comparison to {save_path}")
        plt.close()  # Close figure after saving

def plot_metrics_graph(results, base_psnr, base_ssim, save_path=None):
    names = [res['name'] for res in results]
    psnr_scores = [res['psnr'] for res in results]
    ssim_scores = [res['ssim'] for res in results]

    x = np.arange(len(names)) 

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle('Performance Metrics Comparison', fontsize=16)

    # PSNR
    psnr_bars = ax1.bar(x, psnr_scores, color='skyblue', label='Sharpened PSNR')
    ax1.axhline(y=base_psnr, color='r', linestyle='--', label=f'Blurred baseline ({base_psnr:.2f})')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Peak Signal-to-Noise Ratio')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.legend()
    ax1.bar_label(psnr_bars, padding=-3, fmt='%.2f')

    # SSIM
    ssim_bars = ax2.bar(x, ssim_scores, color='salmon', label='Sharpened SSIM')
    ax2.axhline(y=base_ssim, color='r', linestyle='--', label=f'Blurred baseline ({base_ssim:.4f})')
    ax2.set_ylabel('SSIM')
    ax2.set_title('Structural Similarity Index')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.legend()
    ax2.bar_label(ssim_bars, padding=-3, fmt='%.4f')
    ax2.set_ylim([min(base_ssim, min(ssim_scores)) - 0.05, 1.0])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics graph to {save_path}")
        plt.close()  # Close figure after saving


def plot_training_curves(train_losses, val_losses, val_psnrs, save_path=None):
    """Plot training and validation curves for model performance tracking"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Progress', fontsize=16)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PSNR curve
    ax2.plot(epochs, val_psnrs, 'g-', label='Validation PSNR', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Validation PSNR Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    return fig
