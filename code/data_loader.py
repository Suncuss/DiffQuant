import pandas as pd
import torch
import utils
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=utils.resize224_and_to_tensor_transforms):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = self.annotations.iloc[index, 0]
        image = Image.open(img_path)
        label = torch.tensor(float(self.annotations.iloc[index, 1]), dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return (image, label)
  
def load_expanded_dataset_from_csv(csv_file):
    dataset = CustomDataset(csv_file, transform=utils.resize224_and_to_tensor_transforms)
    print(f"Loaded Dataset with size: {len(dataset)}")
    return dataset

def create_data_loaders(train_data, eval_data, batch_size):
    # Create DataLoaders for the training, validation, and test sets
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)
    return train_loader, eval_loader

