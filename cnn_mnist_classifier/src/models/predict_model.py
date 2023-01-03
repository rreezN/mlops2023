import argparse
import pickle
import sys

import click
import torch
import torch.nn as nn
from model import MyAwesomeModel
from torch import optim
from torch.utils.data import DataLoader, Dataset


class dataset(Dataset):
    def __init__(self, images, labels):
        self.data = images.view(-1, 1, 28, 28)
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item].float(), self.labels[item]

    def __len__(self):
        return len(self.data)

@click.group()
def cli():
    pass

@click.command()
@click.argument("model_checkpoint")
@click.argument("test_path")
def evaluate(model_checkpoint, test_path):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)
    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)

    with open(test_path, "rb") as fb:
        test_images, test_labels = pickle.load(fb)
    test_dataset = dataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    with torch.no_grad():
        # set model to evaluation mode
        model.eval()
        # validation pass here
        accuracy = 0
        for images, labels in test_loader:
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy_ = torch.mean(equals.type(torch.FloatTensor))
            accuracy += accuracy_.item()
        accuracy /= len(test_loader)
    print(f'Accuracy: {accuracy}')


cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
