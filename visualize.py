
from utils import (
  np, torch, plt, DEVICE
)

def plot_image_comparison(model, loader, eval_metric, title=""):
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


def display_image_comparison(blurred_img, sharpened_image, mask=None):
    
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