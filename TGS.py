import zipfile
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

# Step 1: Unzip all ZIP files
def unzip_files(zip_paths, extract_to_base='./data'):
    """
    Unzip all provided ZIP files to a specified directory.
    Args:
        zip_paths (list): List of paths to ZIP files.
        extract_to_base (str): Base directory to extract files to.
    """
    if not os.path.exists(extract_to_base):
        os.makedirs(extract_to_base)
    
    for zip_path in zip_paths:
        print(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_base)
        print(f"Extracted to {extract_to_base}")

# List of ZIP files from your screenshot
zip_files = [
    'competition_data.zip',
    'flamingo.zip',
    'test.zip',
    'train.zip'
]

# Unzip all files (ensure these files are in your working directory)
unzip_files(zip_files)

# Step 2: Explore CSV files
def explore_csv(csv_path):
    """
    Load and print basic info about a CSV file.
    Args:
        csv_path (str): Path to the CSV file.
    """
    print(f"\nExploring {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("First few rows:")
    print(df.head())
    return df

# Load and explore CSV files
csv_files = ['depths.csv', 'train.csv', 'sample_submission.csv']
csv_data = {}
for csv_file in csv_files:
    csv_data[csv_file] = explore_csv(csv_file)

# Step 3: Check extracted directories
def list_extracted_dirs(base_path='./data'):
    """
    List directories created after unzipping.
    Args:
        base_path (str): Directory where files were extracted.
    """
    print(f"\nListing directories in {base_path}:")
    for root, dirs, _ in os.walk(base_path):
        for d in dirs:
            print(os.path.join(root, d))

list_extracted_dirs()

# Step 4: Visualize a few images and masks (assuming train.zip contains images and masks)
def visualize_data(image_dir, mask_dir, num_samples=2):
    """
    Display a few images and their corresponding masks.
    Args:
        image_dir (str): Directory containing images.
        mask_dir (str): Directory containing masks.
        num_samples (int): Number of samples to display.
    """
    image_files = sorted(os.listdir(image_dir))[:num_samples]
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file)  # Assumes masks have same names
        
        # Load image and mask
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Plot image
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')
        
        # Plot mask
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Mask')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Assuming train.zip extracts to data/train/images and data/train/masks
# Adjust paths based on actual structure after unzipping
image_dir = './data/train/images'
mask_dir = './data/train/masks'
if os.path.exists(image_dir) and os.path.exists(mask_dir):
    visualize_data(image_dir, mask_dir)
else:
    print(f"Image or mask directory not found. Check extracted structure.")

# Step 5: Basic Dataset class for future training
class SaltDataset(Dataset):
    """
    PyTorch Dataset for loading TGS Salt data.
    """
    def __init__(self, image_dir, mask_dir, image_ids):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ids = image_ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_file = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, img_file)
        mask_path = os.path.join(self.mask_dir, img_file)
        
        # Load and preprocess
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0)  # [1, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0)   # [1, H, W]
        
        return image, mask

# Example: Create a dataset (using train.csv IDs)
train_df = csv_data['train.csv']
image_ids = train_df['id'].tolist() if 'id' in train_df.columns else os.listdir(image_dir)
dataset = SaltDataset(image_dir, mask_dir, image_ids[:10])  # Limit to 10 for testing
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Test the dataloader
for images, masks in dataloader:
    print(f"Batch shape - Images: {images.shape}, Masks: {masks.shape}")
    break
