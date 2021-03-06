"""
Author: u1921415

This file contains the model, configuration, utilities and training functions
necessary for exposure correction of images. 
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
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True


# Set Parameters to use with the neural net

# Chooses the most appropriate device given the machines constraints
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Set path of dataset. Please change as appripriate
TRAIN_DIR = "WM391_PMA_dataset\\training"
VAL_DIR = "WM391_PMA_dataset\\validation"
# Determines how quickly the gradient is travelled for the
# machine learning model
LEARNING_RATE = 1e-4
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
SAVE_MODEL = False
# Toggle displaying the graph of generator loss values
SHOW_LOSS = False
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
        a.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                    max_pixel_value=255.0,),
        # Transform the image to a pytorch tensor.
        ToTensorV2(),
    ]
)

# Transformation for the ground truth images.
transform_ground_truth = a.Compose(
    [
        # Normalize (blur) the image.
        a.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                    max_pixel_value=255.0,),
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
        # Use the generator to create a new image given the
        # varied exposure image.
        y_fake = gen(x)
        # Remove normalisation applied to the image originally.
        y_fake = y_fake * 0.5 + 0.5
        # Save both the generated image and input image for comparison.
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
    # After saving the example images, set the model back into training mode.
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """
    Save the learned weights generated by the model. The saved weights can
    be loaded back into the model to continue training at a later time.
    """
    print("=> Saving checkpoint")
    # Set a data structure to save the model weights to.
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    # Save the weights to the given file name.
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """
    Update the model weights from a previous state saved in the checkpoint file.
    """
    print("=> Loading checkpoint")
    # Load the model weights file onto the device (GPU/CPU).
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    # Unpack the data from the file and update the weights.
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Update the learning rate to latest learning rate.
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class ExposedImageDataset(Dataset):
    """
    Create a pytorch Dataset object.

    Return the poorly exposed image and its corresponding ground truth image.
    This gives the GAN an input image and a target image to work towards for
    every example in the training dataset. Using pytorch dataloaders requries
    overwriting of the datasets __len__ and the __getitem__ methods to return
    the correct size of the dataset and an example from the dataset respectivly.
    """
    def __init__(self,
                 root_dir,
                 transform_both=None,
                 transform_varied_exposure=None,
                 transform_ground_truth=None):
        
        # Set paths to image directories
        self.root_dir = root_dir
        self.variable_exposure_path = os.path.join(root_dir, "INPUT_IMAGES")
        self.ground_truth_path = os.path.join(root_dir, "GT_IMAGES")

        # Initialise transforms to class variables
        self.transform_both = transform_both
        self.transform_varied_exposure = transform_varied_exposure
        self.transform_ground_truth = transform_ground_truth

        # Get the list of file names from the directories
        self.variable_exposure_images = os.listdir(self.variable_exposure_path)
        self.ground_truth_images = os.listdir(self.ground_truth_path)
        
        # Get length of individual dataset classes
        self.variable_exposure_len = len(self.variable_exposure_images)
        self.ground_truth_len = len(self.ground_truth_images)
        
        # Use the variable exposure length since it holds
        # all the training images
        self.length_dataset = self.variable_exposure_len

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        # Modulo input index to prevent an index out of range of the dataset
        index = index % self.length_dataset
        variable_exposure_image = self.variable_exposure_images[index]
        # Floor the ground truth index by 5 since there are 5 exposures
        # for every corresponding ground truth.
        ground_truth_image = self.ground_truth_images[index // 5]

        # Create full path to image
        variable_exposure_image_path = os.path.join(self.variable_exposure_path,
                                                    variable_exposure_image)
        ground_truth_image_path = os.path.join(self.ground_truth_path,
                                               ground_truth_image)

        # Open the image as an RGB numpy array
        variable_exposure_image = np.array(
            Image.open(variable_exposure_image_path).convert("RGB")
        )
        ground_truth_image = np.array(
            Image.open(ground_truth_image_path).convert("RGB")
        )

        # If there's an image transform for both images, apply the transform.
        if self.transform_both:
            augentations = self.transform_both(image=variable_exposure_image,
                                               image0=ground_truth_image)
            variable_exposure_image = augentations["image"]
            ground_truth_image = augentations["image0"]

        # If there's an image transform for the varied exposure image,
        # apply the transform.
        if self.transform_varied_exposure:
            variable_exposure_image = (self.transform_varied_exposure\
            (image=variable_exposure_image)["image"])

        # If there's an image transform for the ground truth image,
        # apply the transform.
        if self.transform_ground_truth:
            ground_truth_image = self.transform_varied_exposure\
            (image=ground_truth_image)["image"]

        return variable_exposure_image, ground_truth_image


def test_dataset():
    """
    Simple test function to ensure dataset working correctly.
    This snippet should return 5 image pairs (1 varied exposure and
    1 ground truth).If everything is working correctly, the varied exposure
    and ground truth should produce tensors of size 256x256.
    """
    # Create a dataset object.
    dataset = ExposedImageDataset(
        # Directory of training dataset.
        TRAIN_DIR,
        # Pass in image transforms for both images, just the varied exposure and
        # just the ground truth.
        transform_both=both_transform,
        transform_varied_exposure=transform_varied_exposure,
        transform_ground_truth=transform_ground_truth
        )
    # Create a dataloader from the variable exposure dataset
    loader = DataLoader(dataset, batch_size=5)
    # Simple counter to track number of examples printed.
    count = 0
    # Get a varied exposure image and respective ground truth image
    # from dataloader.
    for x, y in loader:
        if(count < 5):
            # Print the shape of both images. They must be the same
            # size (256x256).
            print("Variable exposure: {}".format(x.shape))
            print("Ground truth: {}".format(y.shape))
        else:
            # Once 5 examples printed, stop.
            break
        # Increment the counter on each iteration.
        count+=1


class CNNBlock(nn.Module):
    """
    Convolution block class for discriminator.  
    """
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        # Perform neural net functions in a sequential manner.
        self.conv = nn.Sequential(
            # Apply 2d convolution on each channel of the image.
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False,
                padding_mode="reflect"
            ),
            # Use a batch normalisation function to keep continuity
            # accross the batches.
            nn.BatchNorm2d(out_channels),
            # Use a leakyReLU function for the layer activation.
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        # Move onto the next node in the network by applying the convolution.
        return self.conv(x)


class Discriminator(nn.Module):
    """
    Model for the Discriminator. The model is used for determining whether
    and image is real or fake.
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        # Create a node for the first layer of the network.
        self.initial = nn.Sequential(
            # Perform convolution on the first layer of the network.
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            # Use a leakyReLU function for the layer activation.
            nn.LeakyReLU(0.2),
        )

        # Create empty list to save the results of convolution.
        layers = []
        # Set the channels to the first feature layer.
        in_channels = features[0]
        # Iterate through the remaining features.
        for feature in features[1:]:
            # Append the convolution of each layer to the list.
            layers.append(
                CNNBlock(
                    in_channels, feature, stride=1 if \
                    feature == features[-1] else 2
                ),
            )
            # Set the feature to the next feature.
            in_channels = feature
        # For the first feature, append and convolve the image.
        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1,
                padding_mode="reflect"
            ),
        )
        # Transform the list of conv layers into a sequence of nodes.
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        # Concatenate the prediction and PatchGAN in one dimension
        x = torch.cat([x, y], dim=1)
        # Send the prediction through the initial layer and then
        # the rest of the model.
        x = self.initial(x)
        x = self.model(x)
        return x


def test_discriminator():
    """
    Simple test function for discriminator. Tests that the
    shape of the procued PatchGAN is the correct shape following
    the convolution layers.
    """
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x, y)
    # print(model)
    print(preds.shape)


class Block(nn.Module):
    """
    Convolution block for generator model. Creates a generic
    convoultion that can be used for downscaling and upscaling
    an image through the U-net.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 down=True,
                 act="relu",
                 use_dropout=False):
        super(Block, self).__init__()
        # Perform a sequence of neural net image transformations
        self.conv = nn.Sequential(
            # Convolve the image channel.
            nn.Conv2d(in_channels,
                      out_channels,
                      4,
                      2,
                      1,
                      bias=False,
                      padding_mode="reflect")
            if down
            # If downscaling, use conv2d however
            # if upscaling transpose instead.
            else nn.ConvTranspose2d(in_channels,
                                    out_channels,
                                    4,
                                    2,
                                    1,
                                    bias=False),
            # Use a batch normalisation function to keep continuity
            # accross the batches.
            nn.BatchNorm2d(out_channels),
            # Vary activation function based on input parameters.
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        # Instanciate boolean dropout toggle.
        self.use_dropout = use_dropout
        # Set the dropout rate at 0.5 (50%).
        self.dropout = nn.Dropout(0.5)
        # Instanciate boolean for if downscaling.
        self.down = down

    def forward(self, x):
        # Perform the covultion block to forward to the next node.
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    """
    Model for the generator. A U-net is used to decode and encode the image
    from 256x256 to 1x1 then back to a 256x256 image.
    """
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        # Initial convolution down the U-net.
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        # Sequential convolution blocks to shrink the image downwards to 1x1.
        self.down1 = Block(features, features * 2, down=True, act="leaky",
                           use_dropout=False)
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky",
            use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky",
            use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky",
            use_dropout=False
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky",
            use_dropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky",
            use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        # Sequential convolution blocks to transpose the image back to 256x256.
        self.up1 = Block(
            features * 8, features * 8, down=False, act="relu",
            use_dropout=True
        )
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu",
            use_dropout=True
        )
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu",
            use_dropout=True
        )
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu",
            use_dropout=False
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu",
            use_dropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu",
            use_dropout=False
        )
        self.up7 = Block(
            features * 2 * 2, features, down=False, act="relu",
            use_dropout=False
        )
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4,
                               stride=2, padding=1),
            # Use tanh as the activation function for the output of the U-net.
            nn.Tanh(),
        )

    def forward(self, x):
        # Feed the previous layer of the U-net into the next.
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))


def test_generator():
    """
    Simple test function for generator. Output should produce a prediciton of
    shape 256x256.
    """
    x = torch.randn((1, 3, 256, 256))
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)


def train(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce,
          g_scaler, d_scaler, gen_loss):
    # Creates a progress bar to visualise the iterations over the dataset.
    loop = tqdm(loader, leave=True)
    for idx, (x, y) in enumerate(loop):
        # Send input images to device.
        x = x.to(DEVICE)
        # Send ground truth to device.
        y = y.to(DEVICE)

        # Automatically cast datatypes allowing mixed precision datatypes
        with torch.cuda.amp.autocast():
            # Use the generator to create an image.
            y_fake = gen(x)
            # Allow the discriminator train using the input image and the
            # ground truth.
            D_real = disc(x, y)
            # Apply binary cross entropy loss to determine a probability
            # for how 'real' the image is.
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            # Allow the discriminator train using the generated image
            # and the input.
            D_fake = disc(x, y_fake.detach())
            # Apply bce again between generated and input image.
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            # Average the loss accross input vs target and generated vs target.
            D_loss = (D_real_loss + D_fake_loss) / 2

        # Sets the optimiser gradient to zero.
        disc.zero_grad()
        # Compute the backwards gradient based on the discriminator loss.
        d_scaler.scale(D_loss).backward()
        # Update the parameters of the optimiser based on the gradient descent.
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Automatically cast datatypes allowing mixed precision datatypes.
        with torch.cuda.amp.autocast():
            # Calculate the probability that the generated image appears real.
            D_fake = disc(x, y_fake)
            # Perform binary cross entropy generated and input image.
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            # Calculate the absolute error loss (L1) for the generator.
            L1 = l1_loss(y_fake, y) * L1_LAMBDA
            # Sum the loss of the dicriminator and the error loss.
            G_loss = G_fake_loss + L1

        # Sets the generator gradient to zero.
        opt_gen.zero_grad()
        # Compute the backwards gradient based on the discriminator
        # and generator loss.
        g_scaler.scale(G_loss).backward()
        # Update the parameters of the optimiser based on the gradient descent.
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            # Every tenth iteration, apply a sigmoid function to
            # the discriminator values.
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )
    # After each batch, append the loss to the generator loss.
    gen_loss.append(G_loss.detach().to('cpu'))


def main():
    # Instanciate the discriminator and generator on the device.
    disc = Discriminator(in_channels=3).to(DEVICE)
    gen = Generator(in_channels=3, features=64).to(DEVICE)
    # Instaciate the optimisers for the discriminator and generator.
    opt_disc = optim.Adam(disc.parameters(),
                          lr=LEARNING_RATE,
                          betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(),
                         lr=LEARNING_RATE,
                         betas=(0.5, 0.999))
    # Instanciate the loss functions to local variable.
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if LOAD_MODEL:
        # Load pre trained weights for the generator.
        load_checkpoint(
            CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE,
        )
        # Load pre trained weights for the discriminator.
        load_checkpoint(
            CHECKPOINT_DISC, disc, opt_disc, LEARNING_RATE,
        )

    # Instanciate a training dataset and pass in the paths
    # and image transformations.
    train_dataset = ExposedImageDataset(
        root_dir=TRAIN_DIR,
        transform_both=both_transform,
        transform_varied_exposure=transform_varied_exposure,
        transform_ground_truth=transform_ground_truth
    )
    # Use the dataset to create a dataloader to feed images into the model.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    # Create a gradient scaler for the generator and discriminator.
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    # Using the same approach, create a validation dataset and dataloader.
    val_dataset = ExposedImageDataset(
        root_dir=VAL_DIR,
        transform_both=both_transform,
        transform_varied_exposure=transform_varied_exposure,
        transform_ground_truth=transform_ground_truth
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    # Create an empty list to save loss values for evaluation.
    gen_loss = []

    # Train the model for the specified number of epochs.
    for epoch in range(NUM_EPOCHS):
        train(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler,
            d_scaler, gen_loss
        )

        # If showloss is true..
        if SHOW_LOSS and epoch % 20 == 0:
            # ..plot a graph of loss vs epoch every 20 epochs.
            plt.plot(gen_loss)
            plt.xlabel("Number of epochs")
            plt.ylabel("Generator Loss")
            plt.show()

        # If save model is toggled, save the model weights after every epoch.
        if SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=CHECKPOINT_DISC)

        # At the end of every epoch, save an example image to evaluate
        # the model while it trains.
        save_examples(gen, val_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    main()
