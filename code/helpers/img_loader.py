import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
import torchvision.transforms.functional as TF 

# Define the number of classes for each categorical variable
NUM_SWING_MODE_CLASSES = 10
NUM_GENDER_CLASSES = 2
NUM_HAND_CLASSES = 2
NUM_YEARS_CLASSES = 3
NUM_LEVEL_CLASSES = 4 

class ImgTimeSeriesDataset(Dataset):
    """
    A PyTorch Dataset for loading time series data represented as sequences of 1-bit images,
    along with associated metadata for training or evaluation.

    Args:
        csv_file (str): Path to the CSV file containing metadata and file references.
        root_dir (str): Root directory where the image data is stored.
        split (str): The dataset split, e.g., 'train', 'val', or 'test'.
        transform (callable, optional): A function/transform that takes in a 6-channel
                                       image tensor (with values 0.0 or 1.0)
                                       and returns a transformed version.
                                       Default: None.
    """
    def __init__(self, csv_file: str, root_dir: str, split: str, transform=None):
        # Load the metadata from the CSV file
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        # A predefined list of image filenames expected for each sample
        self.img_filenames = ['Ax.png', 'Ay.png', 'Az.png', 'Gx.png', 'Gy.png', 'Gz.png']

        # Validate split value
        if split not in ['train', 'val', 'test']:
            print(f"Warning: Unrecognized split value '{split}'. Expected 'train', 'val', or 'test'.")


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, index: int):
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple or torch.Tensor:
                - If split is 'train' or 'val', returns a tuple (image_tensor, label_tensor).
                - Otherwise (e.g., 'test' split), returns only the image_tensor.
        """
        # Retrieve unique identifier and swing information for the current sample
        uid = str(self.df['unique_id'].iloc[index]) # Ensure UID is a string for path joining
        swing_folder_name = str(self.df['swing'].iloc[index]) # Ensure swing is a string

        # Construct the base path to the directory containing the 6 images for this sample
        img_folder_path = os.path.join(self.root_dir, uid, swing_folder_name)

        img_tensors = []
        for img_filename in self.img_filenames:
            # Construct the full path to the individual image file
            img_path = os.path.join(img_folder_path, img_filename)

            # Open the image using Pillow.
            # For 1-bit images, converting to mode '1' is most explicit.
            # This ensures the image is binarized by PIL.
            image = Image.open(img_path).convert('1')

            # Convert the PIL Image (mode '1') to a PyTorch tensor.
            # TF.to_tensor converts a PIL Image to a torch.FloatTensor
            # of shape (C x H x W) in the range [0.0, 1.0].
            # For mode '1', C=1, and pixel values 0 or 1 become 0.0 or 1.0.
            img_tensor = TF.to_tensor(image)
            img_tensors.append(img_tensor)

        # Stack the list of 6 image tensors (each [1, H, W]) along the channel dimension
        # This creates a single 6-channel tensor of shape [6, H, W].
        imgs_tensor = torch.cat(img_tensors, dim=0)

        # Apply transformations if any are provided
        if self.transform:
            imgs_tensor = self.transform(imgs_tensor)

        # Prepare output based on the dataset split
        if self.split == 'train' or self.split == 'val':
            # Retrieve raw label values and ensure they are integers for indexing
            swing_mode_val = int(float(self.df['mode'].iloc[index]) - 1.0) 
            gender_val = int(float(self.df['gender'].iloc[index]) - 1.0)
            hand_val = int(float(self.df['hold racket handed'].iloc[index]) - 1.0)
            years_val = int(self.df['play years'].iloc[index])
            # Level is adjusted: original values 2,3,4,5 map to 0,1,2,3
            level_val = int(float(self.df['level'].iloc[index]) - 2.0)

            swing_mode_idx = torch.tensor(swing_mode_val, dtype=torch.long)

            # Apply one-hot encoding for each label
            swing_mode_one_hot = F.one_hot(swing_mode_idx, num_classes=NUM_SWING_MODE_CLASSES)

            # For the tensors that will be directly concatenated into output_tensor,
            # they need to be at least 1-dimensional.
            # We convert scalar values to 1D tensors by wrapping them in a list.
            gender_tensor = torch.tensor([gender_val], dtype=torch.long)
            hand_tensor = torch.tensor([hand_val], dtype=torch.long)
            years_tensor = torch.tensor([years_val], dtype=torch.long)
            level_tensor = torch.tensor([level_val], dtype=torch.long)

            # Concatenate the specified tensors.
            # The error occurred because the original _idx tensors were 0-dimensional.
            # By making them 1-dimensional (e.g., tensor([value])), concatenation works.
            output_tensor = torch.cat([
                gender_tensor,
                hand_tensor,
                years_tensor,
                level_tensor
            ], dim=0).float()

            return imgs_tensor, swing_mode_one_hot, output_tensor
        else:
            return imgs_tensor, swing_mode_one_hot

if __name__ == "__main__":
    # --- IMPORTANT ---
    # Modify these paths to point to YOUR actual data
    your_csv_file_path = "data/train/labels.csv"
    your_root_image_dir = "data/train/images"
    # Specify the split you want to test (e.g., 'train', 'val', 'test')
    your_split = 'train'

    print(f"Attempting to load dataset with:")
    print(f"  CSV file: {your_csv_file_path}")
    print(f"  Image root directory: {your_root_image_dir}")
    print(f"  Split: {your_split}")

    # 1. Instantiate the Dataset
    print("\n--- Testing ImgTimeSeriesDataset with your data ---")
    try:
        dataset = ImgTimeSeriesDataset(
            csv_file=your_csv_file_path,
            root_dir=your_root_image_dir,
            split=your_split, # Use the specified split
            transform=None # Add any torchvision transforms here if needed
                           # e.g., transform=transforms.Compose([transforms.Resize((32,32))])
        )
    except FileNotFoundError:
        print(f"ERROR: Could not find CSV file '{your_csv_file_path}' or image directory '{your_root_image_dir}'.")
        print("Please ensure the paths are correct and your data is accessible.")
        exit()
    except Exception as e:
        print(f"An error occurred during dataset initialization: {e}")
        exit()


    # 2. Test __len__
    print(f"Dataset length: {len(dataset)}")

    # 3. Test __getitem__ for a few samples
    num_samples_to_test = min(3, len(dataset)) # Test at most 3 samples or dataset length
    if num_samples_to_test == 0:
        print("Dataset is empty or could not be loaded. Cannot test __getitem__.")
    else:
        print(f"\nFetching first {num_samples_to_test} samples:")
        for i in range(num_samples_to_test):
            print(f"\nSample {i}:")
            try:
                if your_split == 'train' or your_split == 'val':
                    image_tensor, label_tensor = dataset[i]
                    print(f"  Image tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
                    print(f"  Label tensor: {label_tensor}, shape: {label_tensor.shape}, dtype: {label_tensor.dtype}")
                else: # For 'test' split
                    image_tensor = dataset[i]
                    print(f"  Image tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")

                # Verify image tensor values are 0.0 or 1.0
                unique_values = torch.unique(image_tensor)
                is_binary = all(val.item() == 0.0 or val.item() == 1.0 for val in unique_values)
                print(f"  Image tensor values are binary (0.0 or 1.0): {is_binary}")
                if not is_binary:
                    print(f"    Unique values found: {unique_values}")

            except FileNotFoundError as e: # Catch file not found for specific images
                print(f"  Error fetching sample {i}: A specific image file was not found. {e}")
            except Exception as e:
                print(f"  Error fetching sample {i}: {e}")
                # You can add more specific error handling or debugging info here if needed
                # For example, print the expected path if it's a file access issue:
                # uid_test = dataset.df['unique_id'].iloc[i]
                # swing_test = dataset.df['swing'].iloc[i]
                # print(f"    Expected image folder for this sample: {os.path.join(your_root_image_dir, str(uid_test), str(swing_test))}")


    # 4. (Optional) Test with DataLoader
    if len(dataset) > 0:
        print("\n--- Testing with DataLoader ---")
        try:
            # Adjust batch_size as needed
            dataloader = DataLoader(dataset, batch_size=2, shuffle=(your_split == 'train')) # Shuffle only for train
            
            for batch_idx, data_batch in enumerate(dataloader):
                if your_split == 'train' or your_split == 'val':
                    images, labels = data_batch
                    print(f"Batch {batch_idx + 1}:")
                    print(f"  Images batch shape: {images.shape}, dtype: {images.dtype}")
                    print(f"  Labels batch shape: {labels.shape}, dtype: {labels.dtype}")
                else: # For 'test' split
                    images = data_batch
                    print(f"Batch {batch_idx + 1}:")
                    print(f"  Images batch shape: {images.shape}, dtype: {images.dtype}")

                if batch_idx >= 1: # Test a couple of batches
                    break
            print("DataLoader test successful for a few batches.")
        except Exception as e:
            print(f"Error during DataLoader test: {e}")

    print("\n--- Test script finished ---")
