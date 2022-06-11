"""
Author: u1921415

This file contains the model, configuration, utilities and training functions necessary for
exposure correction of images. 
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
import albumentations as a
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


# Set Parameters to use with the neural net

# Chooses the most appropriate device given the machines constraints
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Set path of dataset. Please change as appripriate
TRAIN_DIR = "WM391_PMA_dataset\\training"
VAL_DIR = "WM391_PMA_dataset\\validation"
# Determines how quickly the gradient is travelled for the machine learning model
LEARNING_RATE = 2e-4
# Sets the number of images that are sent to the device per iteration
BATCH_SIZE = 64
# Number of cpu threads used
NUM_WORKERS = 8
# Size of the images used to train the model
IMAGE_SIZE = 256
# Specifies the number of channels in the images input to the model
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
# Number of times the model is trained with the entire training dataset
NUM_EPOCHS = 300
# Load model weights & parameters from checkpoint state
LOAD_MODEL = False
# Save model weights & parameters to checkpoint file
SAVE_MODEL = True
# Set file location for the discriminator and generator checkpoint files
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"


# Apply a crop to varied exposure and ground truth images
both_transform = a.Compose(
    # Resize to 256x256 pixel image.
    [a.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

# Transformation for the varied exposure iamges.
transform_varied_exposure = a.Compose(
    [
        # Apply a horizonal flip to 50% of the images in the dataset.
        a.HorizontalFlip(p=0.5),
        # Add colour jitter to 20% of the images in the dataset.
        a.ColorJitter(p=0.2),
        # Normalize (blur) the image.
        a.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        # Transform the image to a pytorch tensor.
        ToTensorV2(),
    ]
)

# Transformation for the ground truth images.
transform_ground_truth = a.Compose(
    [
        # Normalize (blur) the image.
        a.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        # Transform the image to a pytorch tensor.
        ToTensorV2(),
    ]
)


def save_examples(gen, val_loader, epoch, folder):
    """
    Use the validation dataset to evaluate the model and save the output of the
    validation image to the given folder name.
    """
    # Get the next variable exposure and corresponding ground truth image from
    # the validation dataset.
    x, y = next(iter(val_loader))
    # Load the images onto correct device.
    x, y = x.to(DEVICE), y.to(DEVICE)
    # Sets the model into inference mode rather than training.
    gen.eval()
    # Disable gradient descent calculation since it is unecessary calculation
    # when performing inference on the model.
    with torch.no_grad():
        # Use the generator to create a new image given the varied exposure image.
        y_fake = gen(x)
        # Remove normalisation applied to the image originally.
        y_fake = y_fake * 0.5 + 0.5
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr