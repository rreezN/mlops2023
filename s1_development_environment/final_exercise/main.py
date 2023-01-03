import argparse
import sys

import torch
import click
from torch import optim
import torch.nn as nn

from data import mnist
from model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 30

    # train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:

            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            print(f'Epoch: {e}, Accuracy: {accuracy.item() * 100}%')
    torch.save(model.state_dict(), f'checkpoint{e}.pth')


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    _, test_set = mnist()
    with torch.no_grad():
        # set model to evaluation mode
        model.eval()
        # validation pass here
        accuracy = 0
        for images, labels in test_set:
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy_ = torch.mean(equals.type(torch.FloatTensor))
            accuracy += accuracy_.item()
        accuracy /= len(test_set)
    print(f'Accuracy: {accuracy}')

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
