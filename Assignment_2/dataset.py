from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class ExposedImageDataset(Dataset):
    def __init__(self, root_varied_exposure, root_ground_truth, transform=None):
        self.root_exposure = root_varied_exposure
        self.root_ground_truth = root_ground_truth
        self.transform = transform

        # Get the file names from the directories
        self.variable_exposure_images = os.listdir(root_varied_exposure)
        self.ground_truth_images = os.listdir(root_ground_truth)
        
        # Get length of individual dataset classes
        self.variable_exposure_len = len(self.variable_exposure_images)
        self.ground_truth_len = len(self.ground_truth_images)
        
        # Use the ground truth length since it has only one of every image
        self.length_dataset = self.ground_truth_len


    def __len__(self):
        return self.length_dataset

    def compare_indices(self, index):
        variable_exposure_index = index % self.variable_exposure_len
        ground_truth_index = (index % self.ground_truth_len) // 5
        return variable_exposure_index, ground_truth_index

    def get_file_name(self, index):
        # Modulo input index to prevent an index out of range of the dataset
        index = index % self.length_dataset
        variable_exposure_image = self.variable_exposure_images[index]
        # Floor the ground truth index by 5 since there are 5 exposures for every corresponding ground truth
        ground_truth_image = self.ground_truth_images[index // 5]

        return variable_exposure_image, ground_truth_image

    def __getitem__(self, index):
        variable_exposure_image, ground_truth_image = self.get_file_name(index)

        # Create full path to image
        variable_exposure_path = os.path.join(self.root_exposure, variable_exposure_image)
        ground_truth_path = os.path.join(self.root_ground_truth, ground_truth_image)

        # Open the image as an RGB numpy array
        variable_exposure_image = np.array(Image.open(variable_exposure_path).convert("RGB"))
        ground_truth_image = np.array(Image.open(ground_truth_path).convert("RGB"))

        if self.transform:
            augentations = self.transform(image=variable_exposure_image, image0=ground_truth_image)
            variable_exposure_image = augentations["image"]
            ground_truth_image = augentations["image0"]

        
        return variable_exposure_image, ground_truth_image

if __name__ == "__main__":
    A = ExposedImageDataset("WM391_PMA_dataset\\training\\INPUT_IMAGES", "WM391_PMA_dataset\\training\\GT_IMAGES")
    test_index = 200000
    image0, image1 = A.get_file_name(test_index)
    print(image0,image1)
    image2, image3 = A.compare_indices(test_index)
    print(image2, image3)
    image4, image5 = A.__getitem__(test_index)
    print(image4, image5)
