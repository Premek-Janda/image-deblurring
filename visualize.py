
from utils import (
    DEVICE, BLUR_KERNEL_SIZE,
    os, np, plt,
    nn, torch, PeakSignalNoiseRatio,
    load_blurred_image, load_grayscale
)
from models import DeblurringSimple, UNet

def plot_one_image_comparison(model, loader, eval_metric, title=""):
    model.eval()
    data_blur, data_sharp = next(iter(loader))
    data_blur = data_blur.to(DEVICE)
    data_sharp = data_sharp.to(DEVICE)
    
    with torch.no_grad():
        deblurred = model(data_blur)
        
    # input vs target
    eval_input = eval_metric(data_blur[0].unsqueeze(0), data_sharp[0].unsqueeze(0)).item()
    # output vs target
    eval_output = eval_metric(deblurred[0].unsqueeze(0), data_sharp[0].unsqueeze(0)).item()
    
    def prepare_img(tensor):
        img = tensor[0].cpu().permute(1, 2, 0).numpy()
        return np.clip(img, 0, 1)

    blur_img = prepare_img(data_blur)
    deblur_img = prepare_img(deblurred)
    sharp_img = prepare_img(data_sharp)


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

def plot_kernel_comparison(model):
    # this only works for the simple convolution model
    if hasattr(model, 'convolution'):
        # access the underlying Conv2d layer
        weights = model.convolution.weight.data[0, 0].cpu().numpy()
        
        plt.figure(figsize=(4, 4))
        plt.imshow(weights, cmap='viridis')
        plt.colorbar()
        plt.title("Learned Kernel (Channel 0)")
    else:
        print("Model does not have a single convolution layer to inspect.")


def compare_models(test_loader):
    print("Loaded last best model")
    
    # load images
    data_blur, data_sharp = next(iter(test_loader))
    data_blur = data_blur.to(DEVICE)
    data_sharp = data_sharp.to(DEVICE)
    
    for file in os.listdir('.'):
        if file.endswith(".pth"):
            # TODO load model 
            model_name = file.replace("best_model_", "").replace(".pth", "")
            model = globals()[model_name]().to(DEVICE)
            model.load_state_dict(torch.load(file, map_location=DEVICE))
            eval_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
            outputs = model(data_blur)
            eval_res = eval_metric(outputs, data_sharp).item()
            # evaluate model
            print(f"Model {model.__class__.__name__} PSNR: {eval_res:.2f}")
    
    # visual Analysis
    plot_kernel_comparison(model)


def display_image_comparison(blurred_img, sharpened_image, mask=None):
    """Displays the original blurry image, the sharpened image, and optionally the edge mask."""
    
    # display
    _, axes = plt.subplots(1, 2 if mask is None else 3, figsize=(18, 6))

    axes[0].imshow(blurred_img, cmap='gray')
    axes[0].set_title('Blurry original')
    axes[0].axis('off')

    axes[1].imshow(sharpened_image, cmap='gray')
    axes[1].set_title('Sharpened')
    axes[1].axis('off')

    if mask is not None:
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title('Edge Mask')
        axes[2].axis('off')
        
        
        
## for traditional methods
def comparison_table(results):
    print(f"{'Method':<25} | {'PSNR (dB)':<12} | {'SSIM':<12}")
    for res in results:
        print(f"{res['name']:<25} | {res['psnr']:<12.2f} | {res['ssim']:<12.4f}")

def plot_image_comparison(results, filename="dandelion.jpg", blur_kernel_size=BLUR_KERNEL_SIZE):
    original_img = load_grayscale(filename)
    blurred_img = load_blurred_image(filename, blur_kernel_size)

    num_images = len(results) + 2
    cols = 3
    rows = (num_images + cols) // cols
    plt.figure(figsize=(12, 3 * rows))
    plt.suptitle("Visual Comparison of Deblurring Methods", fontsize=16)

    plt.subplot(rows, cols, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(rows, cols, 2)
    plt.imshow(blurred_img, cmap='gray')
    plt.title("Blurred")
    plt.axis('off')

    for i, res in enumerate(results):
        plt.subplot(rows, cols, i + 4)
        plt.imshow(res['image'], cmap='gray')
        plt.title(f"{res['name']}\nPSNR: {res['psnr']:.2f}, SSIM: {res['ssim']:.4f}")
        plt.axis('off')

    plt.tight_layout()

def plot_metrics_graph(results, base_psnr, base_ssim):
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
