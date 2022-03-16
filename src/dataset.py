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
    def __init__(self, monet: bool, data_dir: str, transforms: Compose = None):
        super().__init__()
        self.__monet = monet
        self.__data_dir = data_dir
        dataset_type = 'monet' if monet else 'photo'
        self.__dataset_dir = os.path.join(data_dir, dataset_type)
        self.__img_paths = [os.path.join(self.__dataset_dir, img_name)
                            for img_name in os.listdir(self.__dataset_dir)]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.__img_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        # Read image
        img_path = self.__img_paths[index]
        tensor_img = read_image(img_path, torchvision.io.ImageReadMode.RGB)

        # Apply custom transforms
        if self.transforms:
            tensor_img = self.transforms(tensor_img)

        # Return the transformed img
        return tensor_img


def download_dataset(data_dir: str):
    # Skip if dataset is already present
    if os.path.exists(data_dir):
        print(f'Dataset already exists at {data_dir}')
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

    # Make new data dirs
    os.mkdir(os.path.join(data_dir, 'monet'))
    os.mkdir(os.path.join(data_dir, 'photo'))

    # Extract data
    with zipfile.ZipFile(os.path.join(data_dir, f'{dataset}.zip'), 'r') as zip_file:
        zip_file.extractall(os.path.join(data_dir, 'temp'))

    # Remove temporary zip
    os.remove(os.path.join(data_dir, f'{dataset}.zip'))

    # Move contents to monet dir
    monet_temp_dir = os.path.join(data_dir, 'temp', 'monet_jpg')
    for monet_img in os.listdir(monet_temp_dir):
        shutil.move(
            os.path.join(monet_temp_dir, monet_img),
            os.path.join(data_dir, 'monet')
        )

    # Move contents to photo dir
    photo_temp_dir = os.path.join(data_dir, 'temp', 'photo_jpg')
    for photo_img in os.listdir(photo_temp_dir):
        shutil.move(
            os.path.join(photo_temp_dir, photo_img),
            os.path.join(data_dir, 'photo')
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
