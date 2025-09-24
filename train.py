import torch
from configs.default_config import config
from src.dataset import get_dataloaders
from src.model import UNet
from src.trainer import Trainer
from src.visualize import visualize_model_predictions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = get_dataloaders(config)
    model = UNet(**config["model"]).to(device)

    # Add positive class weight to balance ships vs water
    pos_weight = torch.tensor([5.0], device=device)  # adjust 3–10 depending on imbalance

    # Pass it into Trainer

    trainer = Trainer(model, train_loader, val_loader, config, device, use_dice_only=True)

    trainer.train()
    trainer.plot_history()
    visualize_model_predictions(model, val_loader, device, n_batches=1, threshold=0.1, n_images=3)

if __name__ == "__main__":
    main()
