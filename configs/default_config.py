# configs/default_config.py

config = {
    "data": {
        "root": "E:/EAGLE/Second_Semester/Applied_Deep_Learning/ship-detection/data/Patches",  # Base folder for dataset
        "img_dir": "images",                                                                   # Subfolder containing input images
        "mask_dir": "masks",                                                                   # Subfolder containing ground-truth masks
        "patch_size": 64,                                                                      # Size of image patches (64x64 pixels)
        "batch_size": 16,                                                                      # Number of images processed during training
        "num_workers": 4,                                                                      # Number of CPU used for data loading
        "val_split": 0.2,                                                                      # Fraction of dataset used for validation (20%)
    },
    "training": {
        "epochs": 50,                                                                          # Number of times the model will see the entire dataset
        "lr": 1e-3,                                                                            # Learning rate for optimizer
        "optimizer": "adam",                                                                   # Optimization algorithm (Adam)
        "scheduler": "cosine",                                                                 # Learning rate scheduler type (cosine decay)
        "loss": "bce_dice",                                                                    # Loss function (Binary Cross-Entropy + Dice Loss)-suitable for segmentation tasks with imbalanced data.
        "weight_decay": 0.0                                                                    # L2 regularization term to prevent overfitting
    },
    "model": {
        "in_channels": 2,                                                                      # Number of channels in input images, which are 2 a-VV and VH
        "out_channels": 1,                                                                     # Number of channels in output masks which is 1- (binary mask)
        "features": [32, 64, 128],                                                             # Number of filters in each layer of the U-Net
    },
    "logging": {
        "save_dir": "checkpoints",                                                             # Folder to save trained model checkpoints
        "log_interval": 50,                                                                    # Print training info every 50 batches
    }
}
