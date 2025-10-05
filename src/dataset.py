
import os, glob                                                                # For file path operations and listing files
import torch                                                                   # PyTorch library
import numpy as np                                                             # For numerical operations on arrays
from torch.utils.data import Dataset, DataLoader                               # For creating custom datasets and batching
import rasterio                                                                # For reading raster (satellite) images
import albumentations as A                                                     # For image augmentations

# Custom Dataset for ship detection
class ShipDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths                                           # List of image file paths
        self.mask_paths = mask_paths                                         # List of corresponding mask file paths
        self.transform = transform                                           # Optional image augmentations

    # Return total number of samples
    def __len__(self):
        return len(self.img_paths)

    # Get one sample (image + mask) by index
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]                                     # Get image file path
        mask_path = self.mask_paths[idx]                                   # Get corresponding mask file path

        # Read SAR image
        with rasterio.open(img_path) as src:
            vv = src.read(1).astype(np.float32)                           # Read VV polarization
            vh = src.read(2).astype(np.float32)                           # Read VH polarization
            img = np.stack([vv, vh], axis=-1)                      # Combine channels: shape (H, W, 2)
            img = np.clip(img, -25, 10)                            # Clip extreme values
            img = (img + 25.0) / 35.0                                     # Normalize to 0–1

        # Read mask image
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)                       # Read mask channel
            if np.max(mask) > 0:
                mask /= np.max(mask)                                   # Normalize mask to 0–1
            mask = np.expand_dims(mask, axis=-1)                       # Add channel dimension: shape (H, W, 1)

        # Apply augmentations if provided
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]                     # Extract augmented image and mask

        # Convert to PyTorch tensors and change shape to (C, H, W) from (H,W,C)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        return img, mask                                              # Return one sample

# Function to create train and validation dataloaders
def get_dataloaders(cfg):
    # Build image and mask directories from config
    img_dir = os.path.join(cfg["data"]["root"], cfg["data"]["img_dir"])
    mask_dir = os.path.join(cfg["data"]["root"], cfg["data"]["mask_dir"])

    # Get sorted list of all image and mask files
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))

    # Split into training and validation sets
    n = len(img_paths)
    split = int((1 - cfg["data"]["val_split"]) * n)
    train_imgs, val_imgs = img_paths[:split], img_paths[split:]
    train_masks, val_masks = mask_paths[:split], mask_paths[split:]

    # Define augmentations for training

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),                                                    # Random horizontal flip
        # A.VerticalFlip(p=0.5),                                                    # Random vertical flip (commented out)
        A.RandomBrightnessContrast(p=0.3),                                          # Random brightness/contrast
        A.Rotate(limit=30, p=0.5),                                                  # Random rotation up to ±30 degrees
        A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-10, 10), p=0.5)  # Random affine transform
    ])
    val_transform = None                                                           # No augmentations for validation

    # Create dataset objects
    train_ds = ShipDataset(train_imgs, train_masks, transform=train_transform)
    val_ds = ShipDataset(val_imgs, val_masks, transform=val_transform)

    # Create PyTorch dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,                                                              # Shuffle training data
        num_workers=cfg["data"]["num_workers"]                                     # Parallel data loading
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,                                                             # Keep validation data in order
        num_workers=cfg["data"]["num_workers"]
    )

    return train_loader, val_loader                                                # Return dataloaders ready for training
