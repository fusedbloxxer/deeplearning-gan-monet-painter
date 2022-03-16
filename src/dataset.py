from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from torchvision.io import read_image
from typing import Tuple
import torchvision
import zipfile
import kaggle
import shutil
import torch
import os


class MonetDataset(Dataset):
    def __init__(self, train: bool, data_dir: str, transforms: Compose = None):
        super().__init__()
        self.__train = train
        self.__data_dir = data_dir
        dataset_type = 'train' if train else 'test'
        self.__dataset_dir = os.path.join(data_dir, dataset_type)
        self.__img_paths = [os.path.join(self.__dataset_dir, img_name)
                            for img_name in os.listdir(self.__dataset_dir)]
        self.__transforms = transforms

    def __len__(self) -> int:
        return len(self.__img_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        # Read image
        img_path = self.__img_paths[index]
        tensor_img = read_image(img_path, torchvision.io.ImageReadMode.RGB)

        # Apply custom transforms
        if self.__transforms:
            tensor_img = self.__transforms(tensor_img)

        # Return the transformed img
        return tensor_img


def download_dataset(data_dir: str):
    # Skip if dataset is already present
    if os.path.exists(data_dir):
        print('dataset already exists')
        return

    # Setup filesystem
    dataset = 'gan-getting-started'

    # Download the dataset
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(
        competition=dataset,
        path=data_dir,
        force=False,
        quiet=False
    )

    # Make new test-train dirs
    os.mkdir(os.path.join(data_dir, 'train'))
    os.mkdir(os.path.join(data_dir, 'test'))

    # Extract data
    with zipfile.ZipFile(os.path.join(data_dir, f'{dataset}.zip'), 'r') as zip_file:
        zip_file.extractall(os.path.join(data_dir, 'temp'))

    # Remove temporary zip
    os.remove(os.path.join(data_dir, f'{dataset}.zip'))

    # Move contents to train
    train_temp_dir = os.path.join(data_dir, 'temp', 'monet_jpg')
    for train_img in os.listdir(train_temp_dir):
        shutil.move(
            os.path.join(train_temp_dir, train_img),
            os.path.join(data_dir, 'train')
        )

    # Move contents to test
    test_temp_dir = os.path.join(data_dir, 'temp', 'photo_jpg')
    for test_img in os.listdir(test_temp_dir):
        shutil.move(
            os.path.join(test_temp_dir, test_img),
            os.path.join(data_dir, 'test')
        )

    # Remove old dirs
    shutil.rmtree(os.path.join(data_dir, 'temp'))


def compute_stats(dataset: Dataset, num_workers: int = 0, batch_size: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    # Create a dataloader over the dataset
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Find mean and std of dataset
    data_mean = torch.zeros(3)
    data_std = torch.zeros(3)

    # Auxiliary data for std & mean computation
    total_squared_sum = torch.zeros(3)
    total_sum = torch.zeros(3)

    # Compute the train dataset stats
    for img_batch in data_loader:
        total_sum += torch.mean(img_batch, dim=(0, 2, 3))
        total_squared_sum += torch.mean(img_batch ** 2, dim=(0, 2, 3))

    # Compute the values
    data_mean = total_sum / len(data_loader)
    data_mean_squared = total_squared_sum / len(data_loader)
    data_std = torch.sqrt(data_mean_squared - data_mean ** 2)
    return data_mean, data_std
