import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchmetrics import PeakSignalNoiseRatio

import os
import random
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Initial Settings ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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
    
    # Encoder (Down-sampling)
    self.inc = DoubleConv(in_channels, 64)
    self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
    self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
    self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
    self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

    # Decoder (Up-sampling)
    self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
    self.conv1 = DoubleConv(1024, 512) # 512 (up) + 512 (skip)
    
    self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    self.conv2 = DoubleConv(512, 256) # 256 (up) + 256 (skip)
    
    self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.conv3 = DoubleConv(256, 128) # 128 (up) + 128 (skip)
    
    self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.conv4 = DoubleConv(128, 64) # 64 (up) + 64 (skip)
    
    # Final layer
    self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
    self.final_activation = nn.Sigmoid() # Force output to [0, 1]

  def forward(self, x):
    # Encoder
    x1 = self.inc(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)
    
    # Decoder com Skip Connections
    x = self.up1(x5)
    x = torch.cat([x4, x], dim=1)
    x = self.conv1(x)
    
    x = self.up2(x)
    x = torch.cat([x3, x], dim=1)
    x = self.conv2(x)
    
    x = self.up3(x)
    x = torch.cat([x2, x], dim=1)
    x = self.conv3(x)
    
    x = self.up4(x)
    x = torch.cat([x1, x], dim=1)
    x = self.conv4(x)
    
    logits = self.outc(x)
    return self.final_activation(logits)
  
class GoProDataset(Dataset):
  def __init__(self, blur_dir, sharp_dir, image_size=256, is_train=True):
    super().__init__()
    self.blur_dir = blur_dir
    self.sharp_dir = sharp_dir
    self.image_size = image_size
    self.is_train = is_train
    
    self.blur_images = sorted(os.listdir(blur_dir))
    self.sharp_images = sorted(os.listdir(sharp_dir))
    
    assert len(self.blur_images) == len(self.sharp_images), \
      "The 'blur' and 'sharp' folders don't have the same number of images"

    self.to_tensor = T.ToTensor()

  def __len__(self):
    return len(self.blur_images)

  def __getitem__(self, idx):
    blur_img_path = os.path.join(self.blur_dir, self.blur_images[idx])
    sharp_img_path = os.path.join(self.sharp_dir, self.sharp_images[idx])
    
    img_blur_pil = Image.open(blur_img_path).convert('RGB')
    img_sharp_pil = Image.open(sharp_img_path).convert('RGB')

    # --- Synchronized Data Augmentation ---
    
    if self.is_train:
      # --- Training: Random Crop + Random Flip ---
      
      # 1. Random Crop
      i, j, h, w = T.RandomCrop.get_params(
        img_blur_pil, output_size=(self.image_size, self.image_size)
      )
      img_blur = TF.crop(img_blur_pil, i, j, h, w)
      img_sharp = TF.crop(img_sharp_pil, i, j, h, w)

      # 2. Random Horizontal Flip
      if random.random() > 0.5:
        img_blur = TF.hflip(img_blur)
        img_sharp = TF.hflip(img_sharp)
    
    else:
      # --- Validation: Only Center Crop ---
      # (No flips to have consistent results)
      
      # Define the CenterCrop transform
      cropper = T.CenterCrop((self.image_size, self.image_size))
      
      # Apply it
      img_blur = cropper(img_blur_pil)
      img_sharp = cropper(img_sharp_pil)

    # Convert to Tensor (normalize to [0, 1])
    img_blur_tensor = self.to_tensor(img_blur)
    img_sharp_tensor = self.to_tensor(img_sharp)
    
    return img_blur_tensor, img_sharp_tensor

# --- Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 2  # Decrease to 2 or 1 if you get "CUDA out of memory" errors
NUM_EPOCHS = 10 # 50-100 is a good start
IMAGE_SIZE = 256 # The size of our crops (patches)

# --- PATHS (Based on your setup) ---
BASE_DIR = "dataset/Gopro"
BLUR_DIR_TRAIN = os.path.join(BASE_DIR, "train/blur")
SHARP_DIR_TRAIN = os.path.join(BASE_DIR, "train/sharp")
BLUR_DIR_VAL = os.path.join(BASE_DIR, "test/blur")
SHARP_DIR_VAL = os.path.join(BASE_DIR, "test/sharp")

MODEL_SAVE_PATH = "best_deblur_unet_gopro.pth" # We'll save the *best* model

# --- Validation Function ---
# This function will be called at the end of each epoch to use the "test directory"
def validate_model(loader, model, loss_fn, psnr_metric, device):
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    total_psnr = 0.0
    
    with torch.no_grad():
        for (data_blur, data_sharp) in loader:
            data_blur = data_blur.to(device)
            data_sharp = data_sharp.to(device)
            
            outputs = model(data_blur)
            
            # Calculate Loss
            loss = loss_fn(outputs, data_sharp)
            total_loss += loss.item()
            
            # Calculate PSNR
            psnr = psnr_metric(outputs, data_sharp)
            total_psnr += psnr.item()
            
    avg_loss = total_loss / len(loader)
    avg_psnr = total_psnr / len(loader)
    
    model.train() # Set model back to training mode
    return avg_loss, avg_psnr

# --- Main Script Start ---
def main():
  # 1. Model
  model = UNet(in_channels=3, out_channels=3).to(device)
  
  # 2. Loss, Optimizer and Metric
  loss_fn = nn.L1Loss() # MAE Loss is good for images
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
  psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device) # Images are [0, 1]

  # 3. DataLoaders
  train_dataset = GoProDataset(
    blur_dir=BLUR_DIR_TRAIN, 
    sharp_dir=SHARP_DIR_TRAIN, 
    image_size=IMAGE_SIZE,
    is_train=True
  )
  train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4, # Speeds up loading
    pin_memory=True
  )
  
  val_dataset = GoProDataset(
    blur_dir=BLUR_DIR_VAL, 
    sharp_dir=SHARP_DIR_VAL, 
    image_size=IMAGE_SIZE,
    is_train=False # IMPORTANT: disables random augmentation
  )
  val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True
  )
  
  print(f"Starting training with {len(train_dataset)} training images and {len(val_dataset)} validation images.")

  # 4. Training Loop
  best_val_psnr = 0.0 # We'll save the model with the best PSNR

  for epoch in range(NUM_EPOCHS):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]", leave=True)
    epoch_loss = 0.0

    for (data_blur, data_sharp) in loop:
      data_blur = data_blur.to(device)
      data_sharp = data_sharp.to(device)
      
      # Forward
      outputs = model(data_blur)
      loss = loss_fn(outputs, data_sharp)
      
      # Backward
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      epoch_loss += loss.item()
      loop.set_postfix(train_loss=loss.item())
    
    avg_train_loss = epoch_loss / len(train_loader)

    # --- Validation at the end of Epoch ---
    val_loss, val_psnr = validate_model(val_loader, model, loss_fn, psnr_metric, device)
    
    print(f"\nEND OF EPOCH {epoch+1}:")
    print(f"  Average Train Loss: {avg_train_loss:.4f}")
    print(f"  Average Val Loss: {val_loss:.4f}")
    print(f"  Val PSNR: {val_psnr:.2f} dB")
    
    # Save the best model
    if val_psnr > best_val_psnr:
      best_val_psnr = val_psnr
      torch.save(model.state_dict(), MODEL_SAVE_PATH)
      print(f"  -> New best PSNR! Model saved at {MODEL_SAVE_PATH}")

  print("Training completed.")

# Execute training
if __name__ == "__main__":
  main()