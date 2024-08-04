import pandas as pd
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Modify the file paths here
csv_file_path = '/content/drive/MyDrive/Colab/ISIC2019/ISIC_2019_Training_GroundTruth.csv'
image_folder_path = '/content/drive/MyDrive/Colab/ISIC2019/ISIC_2019_Training_Input'

data = pd.read_csv(csv_file_path)

# Extract image filenames and labels
image_files = data['image'].values
labels = data.drop(['image', 'UNK'], axis=1).values

# Split the dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(image_files, labels, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)  # 0.25 of 0.8 = 0.2

# Define a custom dataset class
class SkinDiseaseDataset(Dataset):
    def __init__(self, image_files, labels, image_dir, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx] + ".jpg")
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create training, validation, and testing datasets
train_dataset = SkinDiseaseDataset(train_images, train_labels, image_dir = image_folder_path, transform=transform)
val_dataset = SkinDiseaseDataset(val_images, val_labels, image_dir = image_folder_path, transform=transform)
test_dataset = SkinDiseaseDataset(test_images, test_labels, image_dir = image_folder_path, transform=transform)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))