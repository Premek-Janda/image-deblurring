import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

# --- 1. MODEL DEFINITIONS (Must be copied from train.py) ---
# PyTorch needs this to know how to build the model
# before loading the weights (the .pth file)

class DoubleConv(nn.Module):
    """(Convolution -> Batch Norm -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5); x = torch.cat([x4, x], dim=1); x = self.conv1(x)
        x = self.up2(x); x = torch.cat([x3, x], dim=1); x = self.conv2(x)
        x = self.up3(x); x = torch.cat([x2, x], dim=1); x = self.conv3(x)
        x = self.up4(x); x = torch.cat([x1, x], dim=1); x = self.conv4(x)
        logits = self.outc(x)
        return self.final_activation(logits)

# --- 2. INFERENCE FUNCTION ---

def deblur_single_image(model_path, image_path, device, image_size):
    # 1. Load model
    model = UNet(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Evaluation mode!
    
    # 2. Load input image
    img_pil = Image.open(image_path).convert('RGB')
    
    # 3. Transform image
    transform = T.Compose([
        T.Resize((image_size, image_size)), # Use the variable
        T.ToTensor(),
    ])
    
    input_tensor = transform(img_pil).unsqueeze(0).to(device) # [1, 3, H, W]

    # 4. Make prediction
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    # 5. Convert back to Image
    output_tensor = output_tensor.squeeze(0).cpu() # [3, H, W]
    output_tensor = torch.clamp(output_tensor, 0, 1)
    output_img_pil = T.ToPILImage()(output_tensor)
    
    # Resize the original input to the same size for comparison
    return img_pil.resize((image_size, image_size)), output_img_pil

# --- 3. SCRIPT EXECUTION ---

if __name__ == "__main__":
    
    # --- MISSING VARIABLES ---
    
    # CHANGE THIS: Must be THE SAME value you used in train.py
    # (Probably 128, because of your memory)
    IMAGE_SIZE = 128  
    
    # Define o dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Paths ---
    MODEL_PATH = "best_deblur_unet_gopro.pth"
    # Pick any image from your test folder
    TEST_IMAGE_PATH = "dataset/Gopro/test/blur/0.png" 
    
    # --- Execute ---
    print(f"Using device: {device}")
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Testing image: {TEST_IMAGE_PATH}")

    # Pass the variables to the function
    img_input, img_output = deblur_single_image(MODEL_PATH, TEST_IMAGE_PATH, device, IMAGE_SIZE)

    # --- Show results ---
    print("Showing results...")
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].imshow(img_input)
    ax[0].set_title("Input (Blurry)")
    ax[0].axis('off')

    ax[1].imshow(img_output)
    ax[1].set_title("Output (Deblurred by U-Net)")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()