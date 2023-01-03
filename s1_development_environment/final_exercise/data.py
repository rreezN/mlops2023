import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize

def mnist():
    # exchange with the corrupted mnist dataset

    filenames = ["train_0.npz", "train_1.npz", "train_2.npz", "train_3.npz", "train_4.npz"]

    images = []
    labels = []
    for fname in filenames:
        a = np.load(f"../../data/corruptmnist/{fname}")
        images_ = a['images']
        labels_ = a['labels']
        images.append(images_)
        labels.append(labels_)
    train_images = np.concatenate(images)
    train_labels = np.concatenate(labels)

    test = np.load("../../data/corruptmnist/test.npz")
    test_images = test['images']
    test_labels = test['labels']

    class dataset(Dataset):
        def __init__(self, images, labels):
            self.data = torch.from_numpy(images).view(-1, 1, 28, 28)
            self.labels = torch.from_numpy(labels)

        def __getitem__(self, item):
            return self.normalise(self.data[item].float()), self.labels[item]

        def __len__(self):
            return len(self.data)

        def normalise(self, tnsr):
            return (tnsr-tnsr.mean())/tnsr.std()

    train_dataset = dataset(train_images, train_labels)
    test_dataset = dataset(test_images, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    return train_loader, test_loader
