# src/trainer.py

import os
import torch
import matplotlib.pyplot as plt

# Import custom losses and metrics from utils
from src.utils import BCEDiceLoss, DiceLoss, hard_dice, iou_score

class Trainer:
    """
    Trainer class for segmentation tasks:
      - Supports BCE+Dice or Dice-only loss
      - Optional positive class weighting for BCE
      - LR schedulers: ReduceLROnPlateau or CosineAnnealingLR
      - Early stopping based on monitored metric
      - Save best model checkpoint
    """

    def __init__(self, model, train_loader, val_loader, cfg, device, pos_weight=None, use_dice_only=False):
        self.model = model                                                                     # The PyTorch model to train
        self.train_loader = train_loader                                                       # Training dataset loader
        self.val_loader = val_loader                                                           # Validation dataset loader
        self.cfg = cfg                                                                         # Configuration dictionary
        self.device = device                                                                   # 'cpu' or 'cuda'
        self.pos_weight = pos_weight                                                           # Positive class weight for BCE loss
        self.use_dice_only = use_dice_only                                                     # Flag to use Dice loss only

        # Move model to device (GPU/CPU)
        self.model.to(self.device)

        # Loss function
        if self.use_dice_only:
            self.criterion = DiceLoss()                                                        # Use only Dice loss
        else:
            self.criterion = BCEDiceLoss(pos_weight=pos_weight, device=self.device)            # BCE + Dice loss

        # Optimizer
        opt_name = cfg["training"].get("optimizer", "adam").lower()                            # Default to Adam
        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),                                                       # model weights
                lr=cfg["training"]["lr"],                                                      # learning rate from config
                weight_decay=cfg["training"].get("weight_decay", 0.0)                          # optional L2 regularization
            )
        else:
            raise NotImplementedError(f"Optimizer {opt_name} not implemented. Use 'adam'.")

        #  Scheduler
        self.scheduler = None
        sched_type = cfg["training"].get("scheduler", None)
        if sched_type == "cosine":
            T_max = cfg["training"].get("epochs", 30)                                          # Number of epochs for one cosine cycle
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif sched_type in ["plateau", "reduce_on_plateau"]:
            mode = cfg["training"].get("early_stopping", {}).get("mode", "max")                # monitor mode
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=0.5,                                                                    # Reduce LR by half
                patience=3                                                                     # Wait 3 epochs without improvement before reducing
            )

        # Bookkeeping
        self.num_epochs = cfg["training"].get("epochs", 30)                                    # Total training epochs
        self.train_losses = []                                                                 # Track training loss per epoch
        self.val_losses = []                                                                   # Track validation loss
        self.train_dice = []                                                                   # Track training Dice metric
        self.val_dice = []                                                                     # Track validation Dice
        self.train_iou = []                                                                    # Track training IoU metric
        self.val_iou = []                                                                      # Track validation IoU

        # Early Stopping
        es_cfg = cfg["training"].get("early_stopping", {}) if cfg["training"].get("early_stopping") else {}
        self.early_enabled = es_cfg.get("enabled", True)                                      # Enable/disable early stopping
        self.early_monitor = es_cfg.get("monitor", "val_dice")                                # Metric to monitor
        self.early_mode = es_cfg.get("mode", "max")                                           # "max" or "min"
        self.early_patience = es_cfg.get("patience", 5)                                       # Epochs to wait before stopping
        self.early_min_delta = es_cfg.get("min_delta", 1e-4)                                  # Minimum improvement to reset patience

        # Track best metric for early stopping
        if self.early_mode == "max":
            self.best_metric = -float("inf")
        else:
            self.best_metric = float("inf")
        self.no_improve_count = 0

        # Checkpoint directory
        self.ckpt_dir = cfg["logging"].get("save_dir", "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)                                          # Create directory if it doesn't exist

    #  Compute metrics on a dataset
    def _compute_metrics(self, loader):
        """
        Compute average loss, Dice, IoU on a given DataLoader.
        Returns: avg_loss, avg_dice, avg_iou
        """
        self.model.eval()                                                                  # Evaluation mode
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        n = 0                                                                              # Number of batches
        with torch.no_grad():                                                              # Disable gradient computation
            for imgs, masks in loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                preds = self.model(imgs)
                loss = self.criterion(preds, masks)
                total_loss += loss.item()                                                   # Sum batch loss
                total_dice += hard_dice(masks, preds, thr=0.1).item()                       # Dice metric
                total_iou += iou_score(masks, preds, threshold=0.1).item()                  # IoU metric
                n += 1
        if n == 0:
            return 0.0, 0.0, 0.0
        return total_loss / n, total_dice / n, total_iou / n                                # Return averages

    # Training loop
    def train(self):
        """
        Complete training loop with:
          - optimizer updates
          - scheduler stepping
          - early stopping
          - checkpoint saving
        """
        for epoch in range(1, self.num_epochs + 1):
            # Training epoch
            self.model.train()                                                                # Enable dropout/batchnorm
            running_loss = 0.0
            for imgs, masks in self.train_loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                preds = self.model(imgs)
                loss = self.criterion(preds, masks)                                           # Compute loss

                self.optimizer.zero_grad()                                                    # Reset gradients
                loss.backward()                                                               # Backpropagation
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)         # Gradient clipping
                self.optimizer.step()                                                         # Update weights

                running_loss += loss.item()                                                   # Track batch loss

            #  Compute metrics
            train_loss, train_dice, train_iou = self._compute_metrics(self.train_loader)
            val_loss, val_dice, val_iou = self._compute_metrics(self.val_loader)

            # Store metrics history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_dice.append(train_dice)
            self.val_dice.append(val_dice)
            self.train_iou.append(train_iou)
            self.val_iou.append(val_iou)

            # Scheduler step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Step scheduler with monitored metric
                metric = val_loss if self.early_monitor=="val_loss" else val_dice
                self.scheduler.step(metric)
            elif self.scheduler:
                self.scheduler.step()                                                                  # Cosine annealing step

            # Early stopping logic
            current = val_dice if self.early_monitor=="val_dice" else (val_iou if self.early_monitor=="val_iou" else val_loss)
            improved = (current > self.best_metric + self.early_min_delta) if self.early_mode=="max" else (current < self.best_metric - self.early_min_delta)

            if improved:
                self.best_metric = current
                self.no_improve_count = 0
                ckpt_path = os.path.join(self.ckpt_dir, "best_model.pth")
                torch.save(self.model.state_dict(), ckpt_path)                                          # Save best model
                print(f"EarlyStopping: improvement detected (best={self.best_metric:.6f}). Saved model to {ckpt_path}")
            else:
                self.no_improve_count += 1
                print(f"EarlyStopping: no improvement ({self.no_improve_count}/{self.early_patience})")

            #  Epoch summary
            print(f"Epoch {epoch}/{self.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Train IoU: {train_iou:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

            # Stop if early stopping patience exceeded
            if self.early_enabled and self.no_improve_count >= self.early_patience:
                print(f"Stopping early at epoch {epoch} (no improvement in {self.early_patience} epochs).")
                break

    #  Plot training history 
    def plot_history(self):
        """
        Plot Loss, Dice, and IoU metrics over training epochs.
        """
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(15, 4))

        # Plot Loss
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Val Loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Plot Dice
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.train_dice, label="Train Dice")
        plt.plot(epochs, self.val_dice, label="Val Dice")
        plt.title("Dice Coefficient")
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.legend()

        # Plot IoU
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.train_iou, label="Train IoU")
        plt.plot(epochs, self.val_iou, label="Val IoU")
        plt.title("IoU")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.legend()

        plt.tight_layout()
        plt.show()
