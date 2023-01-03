# import argparse
# import sys
#
import click
import torch

# from torch import optim
# import torch.nn as nn
# import pickle
# from torch.utils.data import Dataset, DataLoader
#
from ..models.model import MyAwesomeModel


@click.command()
@click.argument("model_checkpoint")
@click.argument("test_path")
def visualise(model_checkpoint, test_path):
    print("Visualising until hitting the ceiling")
    print(model_checkpoint)

    # Load Model
    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
