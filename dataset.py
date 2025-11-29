### dataset.py - stores the dataset class

from utils import (
  os, random, kagglehub,
  Dataset, Image, T, TF
)

class GoProDataset(Dataset):
  def __init__(self, split='train', image_size=256, augment=True):
    """
    split: 'train' (80%), 'val' (10%), or 'test' (10%)
    """
    super().__init__()
    
    # download/Locate Dataset
    path = kagglehub.dataset_download("rahulbhalley/gopro-deblur")
    base_path = os.path.join(path, "gopro_deblur")
        
    self.blur_dir = os.path.join(base_path, "blur", "images")
    self.sharp_dir = os.path.join(base_path, "sharp", "images")
    
    self.image_size = image_size
    self.augment = augment
    self.split = split
    
    # get all files and Sort them
    all_blur_files = sorted(os.listdir(self.blur_dir))
    all_sharp_files = sorted(os.listdir(self.sharp_dir))
    
    # basic sanity check
    assert len(all_blur_files) == len(all_sharp_files) and len(all_blur_files) > 0, "Empty or mismatched dataset"

    # create Splits (80% Train, 10% Val, 10% Test)
    total_files = len(all_blur_files)
    train_idx = int(0.8 * total_files)
    val_idx = int(0.9 * total_files)
    
    if split == 'train':
        self.blur_images = all_blur_files[:train_idx]
        self.sharp_images = all_sharp_files[:train_idx]
        self.is_train = True
    elif split == 'val':
        self.blur_images = all_blur_files[train_idx:val_idx]
        self.sharp_images = all_sharp_files[train_idx:val_idx]
        self.is_train = False
    elif split == 'test':
        self.blur_images = all_blur_files[val_idx:]
        self.sharp_images = all_sharp_files[val_idx:]
        self.is_train = False
    else:
        raise ValueError("split must be 'train', 'val', or 'test'")

    print(f"[{split.upper()}] Dataset loaded: {len(self.blur_images)} images")
    self.to_tensor = T.ToTensor()

  def __len__(self):
    return len(self.blur_images)

  def __getitem__(self, idx):
    blur_img_path = os.path.join(self.blur_dir, self.blur_images[idx])
    sharp_img_path = os.path.join(self.sharp_dir, self.sharp_images[idx])
    
    img_blur_pil = Image.open(blur_img_path).convert('RGB')
    img_sharp_pil = Image.open(sharp_img_path).convert('RGB')

    # data Augmentation / Preprocessing
    if self.is_train:      
      # random crop
      i, j, h, w = T.RandomCrop.get_params(
        img_blur_pil, output_size=(self.image_size, self.image_size)
      )
      img_blur = TF.crop(img_blur_pil, i, j, h, w)
      img_sharp = TF.crop(img_sharp_pil, i, j, h, w)

      if self.augment:
        # random horizontal flip
        if random.random() > 0.5:
          img_blur = TF.hflip(img_blur)
          img_sharp = TF.hflip(img_sharp)
          
        # random vertical flip
        # if random.random() > 0.5:
        #   img_blur = TF.vflip(img_blur)
        #   img_sharp = TF.vflip(img_sharp)

    else:      
      # Center crop for validation/test
      cropper = T.CenterCrop((self.image_size, self.image_size))
      img_blur = cropper(img_blur_pil)
      img_sharp = cropper(img_sharp_pil)

    img_blur_tensor = self.to_tensor(img_blur)
    img_sharp_tensor = self.to_tensor(img_sharp)
    
    return img_blur_tensor, img_sharp_tensor
  
    
# TODO add more datasets
# path = kagglehub.dataset_download("darthvader4067/hideblur")

class HideBlurDataset():
  def __init__(self, split='train', image_size=256, augment=True):
    """
    split: 'train' (80%), 'val' (10%), or 'test' (10%)
    """
    super().__init__()
    
    # download/Locate Dataset
    path = kagglehub.dataset_download("darthvader4067/hideblur")
    base_path = os.path.join(path, "hideblur")
        
    # TODO
    # self.blur_dir = os.path.join(base_path, "blur", "images")
    # self.sharp_dir = os.path.join(base_path, "sharp", "images")
    
    # self.image_size = image_size
    # self.augment = augment
    # self.split = split
    
    # # get all files and Sort them
    # all_blur_files = sorted(os.listdir(self.blur_dir))
    # all_sharp_files = sorted(os.listdir(self.sharp_dir))
    
    # # basic sanity check
    # assert len(all_blur_files) == len(all_sharp_files) and len(all_blur_files) > 0, "Empty or mismatched dataset"
