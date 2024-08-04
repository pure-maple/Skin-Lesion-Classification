import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Modify here
csv_file_path = '/content/drive/MyDrive/Colab/ISIC2019/ISIC_2019_Training_GroundTruth.csv'
image_folder_path = '/content/drive/MyDrive/Colab/ISIC2019/ISIC_2019_Training_Input'

data = pd.read_csv(csv_file_path)

# Extract image filenames and labels
data['label'] = data.drop(['image', 'UNK'], axis=1).idxmax(axis=1)
image_files = data['image'].values
labels = data['label'].values

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])

# Split the training set into training and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42, stratify=train_data['label'])

# Set minimum and maximum sample sizes
min_samples = 2000
max_samples = 5000

balanced_train_data = pd.DataFrame()

class RandomRotate180:
    def __call__(self, img):
        if random.random() > 0.5:
            return img.rotate(180)
        return img

# Augmentation transforms for training data
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    RandomRotate180(),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.ColorJitter(contrast=(0.5, 1.5))
])

# Common transforms for all data
common_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for label in train_data['label'].unique():
    label_data = train_data[train_data['label'] == label]
    count = len(label_data)

    if count < min_samples:
        n_samples = min(max_samples, max(min_samples, count * 2))
        n_augment = n_samples - count

        augmented_data = []
        for _, row in label_data.iterrows():
            img_name = os.path.join(image_folder_path, row['image'] + ".jpg")
            image = Image.open(img_name).convert('RGB')
            augmented_data.append([row['image'], label])

        for i in range(n_augment):
            img_name = os.path.join(image_folder_path, label_data.iloc[i % count]['image'] + ".jpg")
            image = Image.open(img_name).convert('RGB')
            augmented_image = augmentation_transforms(image)
            augmented_image_name = f"{label_data.iloc[i % count]['image']}_augmented_{i}"
            augmented_image_path = os.path.join(image_folder_path, augmented_image_name + ".jpg")

            # Save the augmented image only if it doesn't already exist
            if not os.path.exists(augmented_image_path):
                augmented_image.save(augmented_image_path)

            augmented_data.append([augmented_image_name, label])

        augmented_df = pd.DataFrame(augmented_data, columns=['image', 'label'])
        label_data = pd.concat([label_data, augmented_df])

    elif count > max_samples:
        n_samples = max(min_samples, int(count * 0.8))
        label_data = resample(label_data, replace=False, n_samples=n_samples, random_state=42)

    balanced_train_data = pd.concat([balanced_train_data, label_data])

# Separate the image filenames and labels of the training set
train_images = balanced_train_data['image'].values
train_labels = pd.get_dummies(balanced_train_data['label']).values

# Separate the image filenames and labels of the validation set
val_images = val_data['image'].values
val_labels = pd.get_dummies(val_data['label']).values

# Separate the image filenames and labels of the test set
test_images = test_data['image'].values
test_labels = pd.get_dummies(test_data['label']).values

# Define the dataset class
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

        return image, torch.tensor(label, dtype=torch.float32)

# Image transformation for training data
train_transform = transforms.Compose([
    augmentation_transforms,
    common_transforms
])

# Create training, validation, and test datasets
train_dataset = SkinDiseaseDataset(train_images, train_labels, image_dir=image_folder_path, transform=train_transform)
val_dataset = SkinDiseaseDataset(val_images, val_labels, image_dir=image_folder_path, transform=common_transforms)
test_dataset = SkinDiseaseDataset(test_images, test_labels, image_dir=image_folder_path, transform=common_transforms)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Check if balanced
train_counts = balanced_train_data['label'].value_counts()
val_counts = val_data['label'].value_counts()
test_counts = test_data['label'].value_counts()

print("Training set label counts:")
print(train_counts)
print("\nValidation set label counts:")
print(val_counts)
print("\nTest set label counts:")
print(test_counts)
