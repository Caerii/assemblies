import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Assuming config.py is in the parent directory
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import config

def get_cifar10_dataloader(batch_size=config.BATCH_SIZE, augment=config.AUGMENT_DATA, shuffle=config.SHUFFLE_DATA):
    """Creates a DataLoader for the CIFAR-10 dataset."""
    transform_list = [transforms.Resize((32, 32)), transforms.ToTensor()]
    if augment:
        transform_list.insert(0, transforms.RandomHorizontalFlip())
    transform = transforms.Compose(transform_list)
    # Use config for data root
    dataset = datasets.CIFAR10(root=config.DATA_ROOT, train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def preprocess_images(images, target_size):
    """Preprocess images by resizing and normalizing."""
    if len(images.shape) == 2:  # Handle flat tensors
        images = images.view(-1, 3, int(images.size(-1) ** 0.5), int(images.size(-1) ** 0.5))  # Assuming CIFAR-10 images

    resized_images = torch.nn.functional.interpolate(
        images, size=target_size, mode="bilinear", align_corners=False
    )

    # Normalize the images
    resized_images = resized_images / 255.0  # Scale pixel values to [0, 1]
    resized_images = resized_images.flatten(start_dim=1)  # Flatten for further processing

    return resized_images 