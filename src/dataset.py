import zipfile
import kaggle
import shutil
import os


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
