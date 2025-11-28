
from utils import (
  torch, plt, DEVICE
)

def plot_image_comparison(model, loader, device=DEVICE, title_prefix=""):
    model.eval()
    data_blur, data_sharp = next(iter(loader))
    data_blur = data_blur.to(device)
    
    with torch.no_grad():
        deblurred = model(data_blur)
    
    blur_img = data_blur[0].cpu().permute(1, 2, 0).numpy()
    sharp_img = data_sharp[0].cpu().permute(1, 2, 0).numpy()
    deblur_img = deblurred[0].cpu().permute(1, 2, 0).numpy()

    plt.figure(figsize=(12, 4))
    plt.suptitle(title_prefix)
    plt.subplot(1, 3, 1)
    plt.imshow(blur_img)
    plt.title("Blurred Input")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(deblur_img)
    plt.title("Model Output")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(sharp_img)
    plt.title("Target Sharp")
    plt.axis('off')
    
    plt.show()

def plot_kernel_comparison(model):
    # this only works for the simple convolution model
    if hasattr(model, 'convolution'):
        print("\nWhat the model learned (First Channel Center Weights):")
        # access the underlying Conv2d layer
        weights = model.convolution.weight.data[0, 0].cpu().numpy()
        print(weights)
        
        plt.figure(figsize=(4, 4))
        plt.imshow(weights, cmap='viridis')
        plt.colorbar()
        plt.title("Learned Kernel (Channel 0)")
        plt.show()
    else:
        print("Model does not have a single convolution layer to inspect.")