
# Libraries

import torch                                                      # PyTorch: Deep learning framework for tensors, GPU acceleration
import torch.nn as nn                                             # Provides neural network building blocks like layers and losses
import torch.nn.functional as F                                   # Provides functional versions of neural net ops (e.g., activations, loss functions)


# Custom Loss Functions


class DiceLoss(nn.Module):                                        # Dice loss is commonly used in segmentation tasks. It measures overlap between predicted and target masks.

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth                                     # Small constant to avoid division by zero

    def forward(self, preds, targets):
        probs = torch.sigmoid(preds)                             # Apply sigmoid to get probabilities from logits
        probs = probs.view(-1)                                   # Flatten predictions
        targets = targets.view(-1)                               # Flatten targets
        intersection = (probs * targets).sum()                   # Overlap between prediction and target
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice  # Dice loss = 1 - Dice coefficient


class BCEDiceLoss(nn.Module):                                   # Combination of Binary Cross-Entropy (BCE) loss and Dice loss. This balances pixel-wise accuracy (BCE) with region overlap (Dice).

    def __init__(self, pos_weight=None, smooth=1e-6, device=None):
        super().__init__()
        # BCE with optional positive class weighting (useful for imbalanced datasets)
        if pos_weight is not None:
            pw = torch.tensor([pos_weight], device=device) if not isinstance(pos_weight, torch.Tensor) else pos_weight.to(device)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)      # BCE with logits handles sigmoid internally
        else:
            self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)                    # Add dice loss

    def forward(self, preds, targets):
        # Total loss = BCE + Dice
        return self.bce(preds, targets) + self.dice(preds, targets)


class FocalLoss(nn.Module):                                   # Focal loss is useful for imbalanced data. It down-weights easy examples and focuses training on hard negatives.

    def __init__(self, gamma=2.0, alpha=1.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma                                   # Controls strength of focusing
        self.alpha = alpha                                   # Balancing factor between classes
        self.reduction = reduction  # 'mean', 'sum', or 'none'

    def forward(self, preds, targets):
        # BCE without reduction (we'll apply focal adjustment first)
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # Probability of correct classification
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss  # Apply focal scaling
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Metrics


def soft_dice(y_true, y_pred_logits, smooth=1e-6):                    # Soft Dice coefficient: uses probabilities instead of hard threshold.

    y_pred = torch.sigmoid(y_pred_logits)                             # Convert logits to probabilities
    intersection = (y_true * y_pred).sum()                            # Overlap
    return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)


def hard_dice(y_true, y_pred_logits, thr=0.1, smooth=1e-6):            # Hard Dice coefficient: uses thresholded predictions.

    y_pred = (torch.sigmoid(y_pred_logits) > thr).float()              # Apply threshold -> binary mask
    intersection = (y_true * y_pred).sum()                             # Overlap
    return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)


def iou_score(y_true, y_pred_logits, threshold=0.1, smooth=1e-6):      # Intersection-over-Union (IoU) metric. Measures how much predicted and target masks overlap relative to union.

    y_pred = (torch.sigmoid(y_pred_logits) > threshold).float()        # Threshold predictions
    y_true_bin = (y_true > 0.5).float()                                # Ensure targets are binary
    intersection = (y_true_bin * y_pred).sum()                         # Intersection
    union = y_true_bin.sum() + y_pred.sum() - intersection             # Union = sum - intersection
    return (intersection + smooth) / (union + smooth)


def normalize_per_channel(img):                                       # Normalize each channel of an image to zero mean and unit variance. img shape: H x W x C

    for c in range(img.shape[2]):
        img[:,:,c] = (img[:,:,c] - img[:,:,c].mean()) / (img[:,:,c].std() + 1e-6)
    return img
