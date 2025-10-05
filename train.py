# train.py

import torch                                                              # PyTorch library (deep learning framework)
from configs.default_config import config                                 # Import training configuration dictionary
from src.dataset import get_dataloaders                                   # Function to create training/validation dataloaders
from src.model import UNet                                                # UNet model architecture for segmentation
from src.trainer import Trainer                                           # Trainer class (handles training loop, loss, metrics, etc.)
from src.visualize import visualize_model_predictions                     # Function to visualize model predictions after training

def main():
    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else fallback to CPU
    print("Using device:", device)                                         # Print which device is being used

    # build dataloaders
    train_loader, val_loader = get_dataloaders(config)                     # Create training and validation dataloaders based on config

    # build model
    model = UNet(**config["model"]).to(device)                             # Initialize UNet model with parameters from config and move it to device (GPU/CPU)

    # compute or set pos_weight for BCE (optional)
    # Example: hard-coded; you can compute this from masks for a better value
    pos_weight = 5.0                                                       # Weight for positive class in BCE loss (to handle class imbalance)
    pos_weight_tensor = torch.tensor([pos_weight], device=device)     # Convert to tensor and move to device

    # Build trainer: pass pos_weight (Trainer will pass it into BCEDiceLoss)
    trainer = Trainer(
        model,                                                             # The model to train (UNet here)
        train_loader,                                                      # Training dataloader
        val_loader,                                                        # Validation dataloader
        config,                                                            # Training configuration dictionary
        device,                                                            # Device to run on (GPU/CPU)
        pos_weight=pos_weight_tensor,                                      # Positive weight tensor for BCE loss
        use_dice_only=False                                                # If False - use BCE + Dice loss; if True - Dice-only loss
    )

    # Start training (will stop early if configured)
    trainer.train()                                                        # Run the training loop (with early stopping and LR scheduler if configured)

    # Plot results
    trainer.plot_history()                                                 # Plot loss, Dice, and IoU history across epochs

    # Visualize a few predictions (loads the model currently in memory - it's the last checkpoint)
    visualize_model_predictions(
        model,                                                             # Model to visualize predictions from
        val_loader,                                                        # Validation dataloader to pick images from
        device,                                                            # Device (CPU/GPU)
        n_batches=1,                                                       # Number of batches to visualize
        threshold=0.1,                                                     # Threshold for binarizing predicted masks
        n_images=3                                                         # Number of images to show
    )

if __name__ == "__main__":                                                 # Python entry-point check
    main()                                                                 # Call the main() function to start training
