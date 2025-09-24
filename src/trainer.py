# Standard library
import os                                                                    # For creating directories and saving model checkpoints

# PyTorch libraries
import torch                                                                 # Main PyTorch library
import matplotlib.pyplot as plt                                              # For plotting training metrics (Loss, Dice, IoU)

# Import custom utilities from your project
from src.utils import BCEDiceLoss, DiceLoss, hard_dice, iou_score
# BCEDiceLoss: Binary Cross-Entropy + Dice Loss (for training)
# DiceLoss: Dice Loss (for training)
# hard_dice: metric using thresholded predictions
# iou_score: Intersection over Union metric

# Define a Trainer class to handle the entire training and validation workflow
class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg, device, pos_weight=None, use_dice_only=False):
        """
        Initializes the trainer
        Args:
            model: PyTorch model (e.g., UNet)
            train_loader: DataLoader for training set
            val_loader: DataLoader for validation set
            cfg: dictionary containing configuration parameters
            device: 'cuda' or 'cpu'
            pos_weight: weight for positive class (for imbalanced BCE)
            use_dice_only: if True, use DiceLoss only, else BCEDiceLoss
        """
        self.model = model                                                    # Neural network model
        self.train_loader = train_loader                                      # Training data loader
        self.val_loader = val_loader                                          # Validation data loader
        self.cfg = cfg                                                        # Configuration dictionary
        self.device = device                                                  # Device to run training

        # Convert pos_weight to a tensor on the correct device
        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight], device=device)

        # Initialize the loss function
        if use_dice_only:
            self.criterion = DiceLoss()                                         # Only Dice loss
        else:
            self.criterion = BCEDiceLoss(pos_weight=pos_weight, device=device)  # BCE + Dice loss

        # Initialize the optimizer (Adam)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["training"]["lr"]                                           # Learning rate from config
        )

        # Initialize learning rate scheduler (optional)
        self.scheduler = None
        if cfg["training"].get("scheduler") == "cosine":                       # Cosine annealing
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg["training"].get("epochs", 30)        # Total epochs
            )
        elif cfg["training"].get("scheduler") == "plateau":                    # Reduce LR on plateau
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
            )

        # Number of epochs to train
        self.num_epochs = cfg["training"].get("epochs", 30)

        # Lists to track training history
        self.train_losses, self.val_losses = [], []                          # Loss per epoch
        self.train_dice, self.val_dice = [], []                              # Dice per epoch
        self.train_iou, self.val_iou = [], []                                # IoU per epoch

        # Track best validation Dice to save best model
        self.best_val_dice = 0.0

    # Internal method to compute metrics on a given loader (train or val)
    def _compute_metrics(self, loader):
        self.model.eval()                                                   # Set model to evaluation mode (no dropout, batchnorm fixed)
        total_loss, total_dice, total_iou = 0, 0, 0                         # Accumulate metrics
        n_batches = 0                                                       # Count number of batches
        with torch.no_grad():                                               # Disable gradient computation
            for imgs, masks in loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)   # Move to GPU/CPU
                preds = self.model(imgs)                                    # Forward pass
                loss = self.criterion(preds, masks)                         # Compute loss
                total_loss += loss.item()                                   # Accumulate loss

                # Compute metrics with a lower threshold (sparse predictions)
                total_dice += hard_dice(masks, preds, thr=0.1).item()       # Dice coefficient
                total_iou += iou_score(masks, preds, threshold=0.1).item()  # IoU metric
                n_batches += 1  # Increment batch count

        # Return average metrics per batch
        if n_batches == 0:
            return 0.0, 0.0, 0.0
        return total_loss / n_batches, total_dice / n_batches, total_iou / n_batches

    # Main training loop
    def train(self):
        self.model.to(self.device)                                        # Move model to GPU/CPU
        for epoch in range(self.num_epochs):                              # Loop over epochs
            self.model.train()                                            # Set model to training mode
            running_loss = 0.0                                            # Track running loss
            for imgs, masks in self.train_loader:                         # Loop over batches
                imgs, masks = imgs.to(self.device), masks.to(self.device) # Move to device
                preds = self.model(imgs)                                  # Forward pass
                loss = self.criterion(preds, masks)                       # Compute loss
                self.optimizer.zero_grad()                                # Reset gradients
                loss.backward()                                           # Backpropagation
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()                                     # Update parameters
                running_loss += loss.item()                               # Accumulate loss

            # Compute metrics on train and validation sets after each epoch
            train_loss, train_dice, train_iou = self._compute_metrics(self.train_loader)
            val_loss, val_dice, val_iou = self._compute_metrics(self.val_loader)

            # Append metrics to tracking lists
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_dice.append(train_dice)
            self.val_dice.append(val_dice)
            self.train_iou.append(train_iou)
            self.val_iou.append(val_iou)

            # Update learning rate scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_dice)                                        # For plateau scheduler, step based on validation metric
            elif self.scheduler:
                self.scheduler.step()                                                # For cosine scheduler, step every epoch

            # Save the best model based on validation Dice
            if val_dice > self.best_val_dice:
                os.makedirs(self.cfg["logging"]["save_dir"], exist_ok=True)          # Create folder if needed
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.cfg["logging"]["save_dir"], "best_model.pth")  # Save model weights
                )
                self.best_val_dice = val_dice                                         # Update best Dice

            # Print epoch metrics
            print(f"Epoch {epoch+1}/{self.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Train IoU: {train_iou:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

    # Plot training history (loss, Dice, IoU)
    def plot_history(self):
        epochs = range(1, self.num_epochs + 1)                                        # X-axis (epoch numbers)
        plt.figure(figsize=(15,4))                                                    # Figure size

        # Plot Loss curve
        plt.subplot(1,3,1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Dice coefficient curve
        plt.subplot(1,3,2)
        plt.plot(epochs, self.train_dice, label='Train Dice')
        plt.plot(epochs, self.val_dice, label='Val Dice')
        plt.title('Dice Coefficient')
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        plt.legend()

        # Plot IoU curve
        plt.subplot(1,3,3)
        plt.plot(epochs, self.train_iou, label='Train IoU')
        plt.plot(epochs, self.val_iou, label='Val IoU')
        plt.title('IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()

        plt.show()                                                          # Display all plots
