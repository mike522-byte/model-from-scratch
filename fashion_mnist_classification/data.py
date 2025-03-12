import numpy as np
from utils import mnist_reader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch
import matplotlib.pyplot as plt

class FashionMNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.reshape(-1, 1, 28, 28)  # Reshape to (N, 1, 28, 28)
        self.labels = labels
        self.images = torch.tensor(self.images, dtype=torch.float32)  # Convert to Tensor
        self.labels = torch.tensor(self.labels, dtype=torch.long)  # Convert to Tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    

def data_loader(batch_size = 32):
    x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    x_train = x_train.astype(np.float32)/255.0
    x_test = x_test.astype(np.float32)/255.0

    random_seed = 72
    np.random.seed(random_seed)

    # 90% training 10% validation
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=random_seed)

    train_dataset = FashionMNISTDataset(x_train, y_train)
    val_dataset = FashionMNISTDataset(x_val, y_val)
    test_dataset = FashionMNISTDataset(x_test, y_test)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = data_loader()
    print(len(train_loader), len(val_loader), len(test_loader))
    for x, y in train_loader:
        print(x.shape, y.shape)
        break
    for x, y in val_loader:
        print(x.shape, y.shape)
        break
    for x, y in test_loader:
        print(x.shape, y.shape)
        break