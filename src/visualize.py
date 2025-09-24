import torch
import matplotlib.pyplot as plt
import numpy as np

# Visualize a batch of images
# Visualize VV/VH channels, ground truth mask, predictions, and overlays
def visualize_batch(images, masks, preds=None, threshold=0.5, n_images=3): # images: torch.Tensor (B, 2, H, W), masks: torch.Tensor (B, 1, H, W). preds: torch.Tensor (B, 1, H, W), optional, threshold: float, binarization threshold for predictions, and n_images: int, number of images to display

    # Move tensors to CPU and convert to numpy
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    if preds is not None:
        preds = preds.cpu().numpy()
        # Apply threshold for converting probabilities to binary masks
        preds = (preds > threshold).astype(np.uint8)

    # show at most n_images from batch
    batch_size = min(n_images, images.shape[0])

    # Loop over selected images
    for i in range(batch_size):
        # Create subplot
        fig, axes = plt.subplots(1, 6 if preds is not None else 3, figsize=(22, 4))

        # 1- VV channel
        axes[0].imshow(images[i, 0], cmap='gray')
        axes[0].set_title("VV")
        axes[0].axis('off')

        # 2- VH channel
        axes[1].imshow(images[i, 1], cmap='gray')
        axes[1].set_title("VH")
        axes[1].axis('off')

        # 3 - Ground truth mask
        axes[2].imshow(masks[i, 0], cmap='gray')
        axes[2].set_title("GT Mask")
        axes[2].axis('off')

        if preds is not None:
            # 4-Predicted mask (binary, after thresholding)
            axes[3].imshow(preds[i, 0], cmap='Reds')
            axes[3].set_title(f"Pred Mask (thr={threshold})")
            axes[3].axis('off')

            # 5- Overlay GT mask on VV
            axes[4].imshow(images[i, 0], cmap='gray')
            axes[4].imshow(masks[i, 0], cmap='Greens', alpha=0.5)
            axes[4].set_title("Overlay GT on VV")
            axes[4].axis('off')

            # Overlay Pred mask on VV
            axes[5].imshow(images[i, 0], cmap='gray')
            axes[5].imshow(preds[i, 0], cmap='Reds', alpha=0.5)
            axes[5].set_title("Overlay Pred on VV")
            axes[5].axis('off')

        plt.tight_layout()
        plt.show()

# Run model and visualize results
def visualize_model_predictions(model, dataloader, device, n_batches=1, threshold=0.5, n_images=3): # model: PyTorch model, dataloader: torch.utils.data.DataLoader, device: torch.device, n_batches: int, number of batches to visualize, threshold: float, binarization threshold, n_images: int, number of images per batch

    model.eval()                                                # Set model to evaluation mode
    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(dataloader):
            if batch_idx >= n_batches:                         # Stop after n_batches
                break
            # Move data to device
            imgs, masks = imgs.to(device), masks.to(device)
            # Forward pass for getting predictions
            preds = model(imgs)
            # Visualize inputs, masks and predictions
            visualize_batch(imgs, masks, preds, threshold, n_images)
