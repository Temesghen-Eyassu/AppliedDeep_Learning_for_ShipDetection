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
        "epochs": 50,                                                                          # Maximum number of epochs
        "lr": 1e-3,                                                                            # Initial learning rate
        "optimizer": "adam",                                                                   # Optimizer type
        "scheduler": "plateau",                                                                # "plateau" or "cosine" or None
        "loss": "bce_dice",                                                                    # Loss choice (uses src.utils.BCEDiceLoss)
        "weight_decay": 0.0,                                                                   # L2 regularization (Adam)
        # Early stopping configuration:
        "early_stopping": {
            "enabled": True,                                                                   # Turn early stopping on/off
            "monitor": "val_dice",                                                             # Which metric to monitor: "val_loss" or "val_dice" or "val_iou"
            "mode": "max",                                                                     # "max" if larger is better (dice, iou); "min" for loss
            "patience": 5,                                                                     # Number of epochs with no improvement to wait before stopping
            "min_delta": 1e-4                                                                  # Minimum change to qualify as improvement
        }
    },
    "model": {
        "in_channels": 2,                                                                      # Number of channels in input images (VV + VH)
        "out_channels": 1,                                                                     # Output mask channels (binary)
        "features": [32, 64, 128],                                                             # U-Net filter sizes
    },
    "logging": {
        "save_dir": "checkpoints",                                                             # Folder to save model checkpoints
        "log_interval": 50,                                                                    # Print training info every N batches
    }
}
