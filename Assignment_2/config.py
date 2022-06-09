import torch
import albumentations as A

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VARIABLE_EXPOSURE = "WM391_PMA_dataset\\training\\INPUT_IMAGES"
GROUND_TRUTH = "WM391_PMA_dataset\\training\\GT_IMAGES"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500


both_transform = A.Compose(
    [A.Resize(width=500, height=500),], additional_targets={"image0": "image"},
)