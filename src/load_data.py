import os
from collections import defaultdict, Counter
from torchvision import datasets, transforms
from torch.utils.data import random_split
from typing import List


def _get_data_transforms(target_size=(224, 224)):
    data_transforms = transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return data_transforms


def load_dataset(data_dir='data', target_size=(224, 224)):
    return datasets.ImageFolder(data_dir, transform=_get_data_transforms(target_size))


def split_dataset(full_dataset, train_size=0.7, val_size=0.15):
    train_size = int(train_size * len(full_dataset))
    val_size = int(val_size * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset


def get_specific_images(base_folder: str) -> List[str]:
    alz_folder = os.path.join(base_folder, 'alzheimers_dataset')
    normal_folder = os.path.join(base_folder, 'normal')

    # List to store the paths of the matched images
    selected_images = []

    # Get images from 'alz' folder that start with 'mild'
    if os.path.exists(alz_folder):
        for file_name in os.listdir(alz_folder):
            if file_name.startswith('mild'):
                selected_images.append(os.path.join(alz_folder, file_name))

    # Get images from 'normal' folder that start with 'non'
    if os.path.exists(normal_folder):
        for file_name in os.listdir(normal_folder):
            if file_name.startswith('non'):
                selected_images.append(os.path.join(normal_folder, file_name))

    return selected_images


def load_specific_dataset(data_dir='data', target_size=(224, 224)):
    '''lädt nur Bilder mit gewissen Präfixen'''

    class FilteredImageFolder(datasets.ImageFolder):
        def _init_(self, root, transform=None, target_transform=None):
            super(FilteredImageFolder, self)._init_(root, transform=transform, target_transform=target_transform)
            # Filtert die Samples (Bilder), die mit 'non' oder 'mild' beginnen
            self.samples = [(path, label) for path, label in self.samples if
                            os.path.basename(path).startswith(('non', 'mild', 'verymild', 'moderate'))]
            self.imgs = self.samples  # Für Kompatibilität mit älteren torchvision-Versionen

    dataset = FilteredImageFolder(data_dir, transform=_get_data_transforms(target_size))

    # Zählt die Anzahl Bilder pro Unterordner
    folder_counts = Counter([os.path.dirname(path).split('/')[-1] for path, _ in dataset.samples])

    # Ausgabe der Anzahl an Bildern in jedem Unterordner
    for folder, count in folder_counts.items():
        print(f"Ordner '{folder}': {count} Bilder")

    return dataset
