import torch                                                              # PyTorch library used for tensor operations, model computations, GPU acceleration
import matplotlib.pyplot as plt                                           # Matplotlib used for plotting images, masks, overlays
import numpy as np                                                        # NumPy used for array manipulations, thresholding, converting tensors to arrays

# Visualize a batch of images
# Shows VV/VH channels, ground truth mask, predicted mask, and overlays
def visualize_batch(images, masks, preds=None, threshold=0.5, n_images=3):
    # images: torch.Tensor (B, 2, H, W) → batch of input images (VV and VH channels)
    # masks: torch.Tensor (B, 1, H, W) → batch of ground truth masks
    # preds: torch.Tensor (B, 1, H, W), optional → batch of predicted masks
    # threshold: float → binarization threshold for predictions
    # n_images: int → number of images from the batch to display

    # Move tensors to CPU and convert to NumPy arrays for plotting
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    if preds is not None:                                                # If predicted masks are provided
        preds = preds.cpu().numpy()                                      # Convert predictions to numpy
        # Apply threshold: probabilities > threshold become 1, else 0
        preds = (preds > threshold).astype(np.uint8)

    # Show at most n_images from the batch
    batch_size = min(n_images, images.shape[0])

    # Loop over each selected image
    for i in range(batch_size):
        # Create subplot: 6 panels if predictions are available, 3 panels otherwise
        fig, axes = plt.subplots(1, 6 if preds is not None else 3, figsize=(22, 4))

        # 1 - VV channel visualization
        axes[0].imshow(images[i, 0], cmap='gray')                        # Display first channel (VV)
        axes[0].set_title("VV")                                          # Title for clarity
        axes[0].axis('off')                                              # Hide axes

        # 2 - VH channel visualization
        axes[1].imshow(images[i, 1], cmap='gray')                        # Display second channel (VH)
        axes[1].set_title("VH")
        axes[1].axis('off')

        # 3 - Ground truth mask
        axes[2].imshow(masks[i, 0], cmap='gray')                          # Display GT mask
        axes[2].set_title("GT Mask")
        axes[2].axis('off')

        if preds is not None:
            # 4 - Predicted mask (binary after thresholding)
            axes[3].imshow(preds[i, 0], cmap='Reds')                      # Pred mask in red
            axes[3].set_title(f"Pred Mask (thr={threshold})")
            axes[3].axis('off')

            # 5 - Overlay GT mask on VV channel
            axes[4].imshow(images[i, 0], cmap='gray')                     # Background VV channel
            axes[4].imshow(masks[i, 0], cmap='Greens', alpha=0.5)         # GT overlayed in green
            axes[4].set_title("Overlay GT on VV")
            axes[4].axis('off')

            # 6 - Overlay Pred mask on VV channel
            axes[5].imshow(images[i, 0], cmap='gray')                     # Background VV channel
            axes[5].imshow(preds[i, 0], cmap='Reds', alpha=0.5)           # Prediction overlayed in red
            axes[5].set_title("Overlay Pred on VV")
            axes[5].axis('off')

        plt.tight_layout()                                                # Adjust spacing between subplots
        plt.show()                                                        # Display the figure


# Run model and visualize results
def visualize_model_predictions(model, dataloader, device, n_batches=1, threshold=0.5, n_images=3):
    # model: PyTorch segmentation model
    # dataloader: DataLoader containing validation or test dataset
    # device: torch.device (CPU or GPU)
    # n_batches: int → number of batches to visualize
    # threshold: float → binarization threshold for predictions
    # n_images: int → number of images per batch to show

    model.eval()                                                         # Set model to evaluation mode (disables dropout, batch norm updates)
    with torch.no_grad():                                                # Disable gradient computation (saves memory & speeds up inference)
        for batch_idx, (imgs, masks) in enumerate(dataloader):
            if batch_idx >= n_batches:                                   # Stop after visualizing n_batches
                break
            # Move inputs and masks to device
            imgs, masks = imgs.to(device), masks.to(device)
            # Forward pass through the model to get predictions
            preds = model(imgs)
            # Call batch visualization function
            visualize_batch(imgs, masks, preds, threshold, n_images)
